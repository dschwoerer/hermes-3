#include "../include/sheath_boundary_parallel.hxx"

#include <bout/output_bout_types.hxx>

#include "bout/constants.hxx"
#include "bout/mesh.hxx"

#include "bout/parallel_boundary_region.hxx"
#include "bout/boundary_iterator.hxx"

using bout::globals::mesh;

namespace {
BoutReal clip(BoutReal value, BoutReal min, BoutReal max) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

BoutReal floor(BoutReal value, BoutReal min) {
  if (value < min)
    return min;
  return value;
}

Ind3D indexAt(const Field3D& f, int x, int y, int z) {
  int ny = f.getNy();
  int nz = f.getNz();
  return Ind3D{(x * ny + y) * nz + z, ny, nz};
}
}


extern Options* tracking;
SheathBoundaryParallel::SheathBoundaryParallel(std::string name, Options &alloptions, Solver *) {
  AUTO_TRACE();
  
  Options &options = alloptions[name];

  Ge = options["secondary_electron_coef"]
           .doc("Effective secondary electron emission coefficient")
           .withDefault(0.0);

  if ((Ge < 0.0) or (Ge > 1.0)) {
    throw BoutException("Secondary electron emission must be between 0 and 1 ({:e})", Ge);
  }
  
  sin_alpha = options["sin_alpha"]
                  .doc("Sin of the angle between magnetic field line and wall surface. "
                       "Should be between 0 and 1")
                  .withDefault(1.0);

  if ((sin_alpha < 0.0) or (sin_alpha > 1.0)) {
    throw BoutException("Range of sin_alpha must be between 0 and 1");
  }

  always_set_phi =
      options["always_set_phi"]
          .doc("Always set phi field? Default is to only modify if already set")
          .withDefault<bool>(false);

  const Options& units = alloptions["units"];
  const BoutReal Tnorm = units["eV"];

  // Read wall voltage, convert to normalised units
  wall_potential = options["wall_potential"]
                       .doc("Voltage of the wall [Volts]")
                       .withDefault(Field3D(0.0))
                   / Tnorm;

  // init parallel bc iterator
  yboundary.init(options);
  // Note: wall potential at the last cell before the boundary is used,
  // not the value at the boundary half-way between cells. This is due
  // to how twist-shift boundary conditions and non-aligned inputs are
  // treated; using the cell boundary gives incorrect results.

  floor_potential = options["floor_potential"]
                        .doc("Apply a floor to wall potential when calculating Ve?")
                        .withDefault<bool>(true);
}

void SheathBoundaryParallel::transform(Options &state) {
  AUTO_TRACE();

  Options& allspecies = state["species"];
  Options& electrons = allspecies["e"];

  // Need electron properties
  // Not const because boundary conditions will be set
  Field3D Ne = toFieldAligned(floor(GET_NOBOUNDARY(Field3D, electrons["density"]), 0.0));
  Field3D Te = toFieldAligned(GET_NOBOUNDARY(Field3D, electrons["temperature"]));
  Field3D Pe = IS_SET_NOBOUNDARY(electrons["pressure"])
    ? toFieldAligned(getNoBoundary<Field3D>(electrons["pressure"]))
    : Te * Ne;

  // Ratio of specific heats
  const BoutReal electron_adiabatic =
      IS_SET(electrons["adiabatic"]) ? get<BoutReal>(electrons["adiabatic"]) : 5. / 3;

  // Mass, normalised to proton mass
  const BoutReal Me =
      IS_SET(electrons["AA"]) ? get<BoutReal>(electrons["AA"]) : SI::Me / SI::Mp;

  // This is for applying boundary conditions
  Field3D Ve = IS_SET_NOBOUNDARY(electrons["velocity"])
    ? toFieldAligned(getNoBoundary<Field3D>(electrons["velocity"]))
    : zeroFrom(Ne);

  bool has_NVe = IS_SET_NOBOUNDARY(electrons["momentum"]);
  Field3D NVe;
  if (has_NVe) {
    NVe = toFieldAligned(getNoBoundary<Field3D>(electrons["momentum"]));
  }

  Coordinates *coord = mesh->getCoordinates();

  //////////////////////////////////////////////////////////////////
  // Electrostatic potential
  // If phi is set, use free boundary condition
  // If phi not set, calculate assuming zero current
  Field3D phi;
  if (IS_SET_NOBOUNDARY(state["fields"]["phi"])) {
    phi = toFieldAligned(getNoBoundary<Field3D>(state["fields"]["phi"]));
  } else {
    // Calculate potential phi assuming zero current
    // Note: This is equation (22) in Tskhakaya 2005, with I = 0

    // Need to sum  s_i Z_i C_i over all ion species
    //
    // To avoid looking up species for every grid point, this
    // loops over the boundaries once per species.
    Field3D ion_sum {zeroFrom(Ne)};
    phi = emptyFrom(Ne); // So phi is field aligned

    // Iterate through charged ion species
    for (auto& kv : allspecies.getChildren()) {
      Options& species = allspecies[kv.first];

      if ((kv.first == "e") or !IS_SET(species["charge"])
          or (get<BoutReal>(species["charge"]) == 0.0)) {
        continue; // Skip electrons and non-charged ions
      }

      const Field3D Ni = toFieldAligned(floor(GET_NOBOUNDARY(Field3D, species["density"]), 0.0));
      const Field3D Ti = toFieldAligned(GET_NOBOUNDARY(Field3D, species["temperature"]));
      const BoutReal Mi = GET_NOBOUNDARY(BoutReal, species["AA"]);
      const BoutReal Zi = GET_NOBOUNDARY(BoutReal, species["charge"]);

      const BoutReal adiabatic = IS_SET(species["adiabatic"])
                                     ? get<BoutReal>(species["adiabatic"])
                                     : 5. / 3; // Ratio of specific heats (ideal gas)

      iter_regions([&](auto& region) {
        for (auto& pnt : region) {
          const auto& i = pnt.ind();
          BoutReal s_i =
              clip(pnt.extrapolate_sheath_o2([&, Ni, Ne](int yoffset, Ind3D ind) {
                return Ni.ynext(yoffset)[ind] / Ne.ynext(yoffset)[ind];
              }),
                   0.0, 1.0);

          if (!std::isfinite(s_i)) {
            s_i = 1.0;
          }
          BoutReal te = Te[i];
          BoutReal ti = Ti[i];

          // Equation (9) in Tskhakaya 2005
          BoutReal grad_ne = pnt.extrapolate_grad_o2(Ne);
          BoutReal grad_ni = pnt.extrapolate_grad_o2(Ni);

          // Note: Needed to get past initial conditions, perhaps
          // transients but this shouldn't happen in steady state
          if (fabs(grad_ni) < 1e-3) {
            grad_ni = grad_ne = 1e-3; // Remove kinetic correction term
          }

          BoutReal C_i_sq =
              clip((adiabatic * ti + Zi * s_i * te * grad_ne / grad_ni) / Mi, 0,
                   100); // Limit for e.g. Ni zero gradient

          // Note: Vzi = C_i * sin(α)
	  BoutReal toadd = s_i * Zi * sin_alpha * sqrt(C_i_sq);
	  if (legacy_match && pnt.dir == 1) {
	    // sin_alpha missing
	    toadd = s_i * Zi * sqrt(C_i_sq);
	  }
          ion_sum[i] += toadd;
        }
      }); // end iter_regions
    }

    phi.allocate();
    phi.splitParallelSlicesAndAllocate();

    // ion_sum now contains  sum  s_i Z_i C_i over all ion species
    // at mesh->ystart and mesh->yend indices
    iter_regions([&](auto& region) {
      for (const auto& pnt : region) {
        auto i = pnt.ind();

	BoutReal thisphi;
	if (Te[i] <= 0.0) {
	  thisphi = 0.0;
	} else {
	  thisphi = Te[i] * log(sqrt(Te[i] / (Me * TWOPI)) * (1. - Ge) / ion_sum[i]);
	}

	thisphi += wall_potential[i];

        pnt.setAll(phi, thisphi);
      }
    }); // end iter_regions
  }

  //////////////////////////////////////////////////////////////////
  // Electrons

  Field3D electron_energy_source = electrons.isSet("energy_source")
    ? toFieldAligned(getNonFinal<Field3D>(electrons["energy_source"]))
    : zeroFrom(Ne);

  iter_regions([&](auto& region) {
    for (const auto& pnt : region) {
      auto i = pnt.ind();

      // Free gradient of log electron density and temperature
      // Limited so that the values don't increase into the sheath
      // This ensures that the guard cell values remain positive
      // exp( 2*log(N[i]) - log(N[ip]) )
      pnt.limitFree(Ne);
      pnt.limitFree(Te);
      pnt.limitFree(Pe);

      // Free boundary potential linearly extrapolated
      const BoutReal phiGradient = pnt.extrapolate_grad_o2(phi);
      pnt.neumann_o1(phi, phiGradient);

      const BoutReal nesheath = pnt.interpolate_sheath_o1(Ne);
      const BoutReal tesheath = pnt.interpolate_sheath_o1(Te);  // electron temperature
      const BoutReal phi_wall = pnt.ythis(wall_potential);

      const BoutReal phisheath = floor_potential ? floor(
            pnt.interpolate_sheath_o1(phi), phi_wall) // Electron saturation at phi = phi_wall
	    : pnt.interpolate_sheath_o1(phi);

      // Electron sheath heat transmission
      const BoutReal gamma_e = floor(2 / (1. - Ge) + (phisheath - phi_wall) / floor(tesheath, 1e-5), 0.0);

      // Electron velocity into sheath (< 0)
      const BoutReal vesheath = (tesheath < 1e-10) ?
          0.0 :
          pnt.dir * sqrt(tesheath / (TWOPI * Me)) * (1. - Ge) * exp(-(phisheath - phi_wall) / tesheath);

      pnt.dirichlet_o2(Ve, vesheath);
      if (has_NVe) {
	pnt.dirichlet_o2(NVe, Me * nesheath * vesheath);
      }

      // Take into account the flow of energy due to fluid flow
      // This is additional energy flux through the sheath
      // Note: sign depends on sign of vesheath
      BoutReal q = ((gamma_e - 1 - 1 / (electron_adiabatic - 1)) * tesheath
                      - 0.5 * Me * SQ(vesheath))
                     * nesheath * vesheath;

      // Multiply by cell area to get power
      const BoutReal flux =
          q * (pnt.ythis(coord->J) + pnt.ynext(coord->J))
          / (sqrt(pnt.ythis(coord->g_22)) + sqrt(pnt.ynext(coord->g_22)));

      // Divide by volume of cell to get energy loss rate (sign depending on vesheath)
      const BoutReal power = flux / (coord->dy[pnt.ind()] * pnt.ythis(coord->J));

#if CHECKLEVEL >= 1
      if (!std::isfinite(power)) {
	throw BoutException("Non-finite power {} at {} : Te {} Ne {} Ve {} phi {}, {} => q {}, flux {}",
			    power, i, tesheath, nesheath, vesheath, phi[i], phisheath, q, flux);
      }
#endif

      electron_energy_source[i] -= pnt.dir * power;
    }
  }); // end iter_regions

  // Set electron density and temperature, now with boundary conditions
  setBoundary(electrons["density"], fromFieldAligned(Ne));
  setBoundary(electrons["temperature"], fromFieldAligned(Te));
  setBoundary(electrons["pressure"], fromFieldAligned(Pe));

  // Add energy source (negative in cell next to sheath)
  // Note: already includes previously set sources
  set(electrons["energy_source"], fromFieldAligned(electron_energy_source));

  if (IS_SET_NOBOUNDARY(electrons["velocity"])) {
    setBoundary(electrons["velocity"], fromFieldAligned(Ve));
  }
  if (has_NVe) {
    setBoundary(electrons["momentum"], fromFieldAligned(NVe));
  }

  if (always_set_phi or IS_SET_NOBOUNDARY(state["fields"]["phi"])) {
    // Set the potential, including boundary conditions
    phi = fromFieldAligned(phi);
    //output.write("-> phi {}\n", phi(10, mesh->yend+1, 0));
    setBoundary(state["fields"]["phi"], phi);
  }

  //////////////////////////////////////////////////////////////////
  // Iterate through all ions
  for (auto& kv : allspecies.getChildren()) {
    if (kv.first == "e") {
      continue; // Skip electrons
    }

    Options& species = allspecies[kv.first]; // Note: Need non-const

    // Ion charge
    const BoutReal Zi =
        IS_SET(species["charge"]) ? get<BoutReal>(species["charge"]) : 0.0;

    if (Zi == 0.0) {
      continue; // Neutral -> skip
    }

    // Characteristics of this species
    const BoutReal Mi = get<BoutReal>(species["AA"]);

    const BoutReal adiabatic = IS_SET(species["adiabatic"])
                                   ? get<BoutReal>(species["adiabatic"])
                                   : 5. / 3; // Ratio of specific heats (ideal gas)

    // Density and temperature boundary conditions will be imposed (free)
    Field3D Ni = toFieldAligned(floor(getNoBoundary<Field3D>(species["density"]), 0.0));
    Field3D Ti = toFieldAligned(getNoBoundary<Field3D>(species["temperature"]));
    Field3D Pi = species.isSet("pressure")
      ? toFieldAligned(getNoBoundary<Field3D>(species["pressure"]))
      : Ni * Ti;

    // Get the velocity and momentum
    // These will be modified at the boundaries
    // and then put back into the state
    Field3D Vi = species.isSet("velocity")
      ? toFieldAligned(getNoBoundary<Field3D>(species["velocity"]))
      : zeroFrom(Ni);
    Field3D NVi = species.isSet("momentum")
      ? toFieldAligned(getNoBoundary<Field3D>(species["momentum"]))
      : Mi * Ni * Vi;

    // Energy source will be modified in the domain
    Field3D energy_source = species.isSet("energy_source")
      ? toFieldAligned(getNonFinal<Field3D>(species["energy_source"]))
      : zeroFrom(Ni);

    iter_regions([&](auto& region) {
      for (const auto& pnt : region) {

        // auto i = pnt.ind();

        // Free gradient of log electron density and temperature
        // This ensures that the guard cell values remain positive
        // exp( 2*log(N[i]) - log(N[ip]) )

        pnt.limitFree(Ni);
        pnt.limitFree(Ti);
        pnt.limitFree(Pi);

        // Calculate sheath values at half-way points (cell edge)
        const BoutReal nesheath = pnt.interpolate_sheath_o1(Ne);
	const BoutReal nisheath = pnt.interpolate_sheath_o1(Ni);
	const BoutReal tesheath = floor(pnt.interpolate_sheath_o1(Te), 1e-5);  // electron temperature
	const BoutReal tisheath = floor(pnt.interpolate_sheath_o1(Ti), 1e-5);  // ion temperature

	// Ion sheath heat transmission coefficient
	// Equation (22) in Tskhakaya 2005
	// with 
	//
	// 1 / (1 + ∂_{ln n_e} ln s_i = s_i ∂_z n_e / ∂_z n_i
	// (from comparing C_i^2 in eq. 9 with eq. 20
	//
	//BoutReal s_i = (nesheath > 1e-5) ? nisheath / nesheath : 0.0; // Concentration ; upper_y
	BoutReal s_i = clip(nisheath / floor(nesheath, 1e-10), 0, 1); // Concentration ; lower_y
	if (legacy_match && pnt.dir == -1){
	  s_i = (nesheath > 1e-5) ? nisheath / nesheath : 0.0;
	}

	BoutReal grad_ne = pnt.extrapolate_grad_o2(Ne);
	BoutReal grad_ni = pnt.extrapolate_grad_o2(Ni);

	if (fabs(grad_ni) < 1e-3) {
	  grad_ni = grad_ne = 1e-3; // Remove kinetic correction term
	}

	// Ion speed into sheath
	// Equation (9) in Tskhakaya 2005
	//
	BoutReal C_i_sq =
	  clip((adiabatic * tisheath + Zi * s_i * tesheath * grad_ne / grad_ni) / Mi,
	       0, 100); // Limit for e.g. Ni zero gradient

	const BoutReal gamma_i = 2.5 + 0.5 * Mi * C_i_sq / tisheath; // + Δγ 

	const BoutReal visheath = pnt.dir * sqrt(C_i_sq); // sign changes -> into sheath

	// Set boundary conditions on flows
	pnt.dirichlet_o2(Vi, visheath);
       	pnt.dirichlet_o2(NVi, Mi * nisheath * visheath);

	// Take into account the flow of energy due to fluid flow
	// This is additional energy flux through the sheath
	// Note: Sign depends on sign of visheath
	BoutReal q =
	  ((gamma_i - 1 - 1 / (adiabatic - 1)) * tisheath - 0.5 * C_i_sq * Mi)
	  * nisheath * visheath;
	if (legacy_match and pnt.dir == -1) {
	  // Mi position switched with C_i_sq
	  q =
              ((gamma_i - 1 - 1 / (adiabatic - 1)) * tisheath - 0.5 * Mi * C_i_sq)
              * nisheath * visheath;
	}

	if (q * pnt.dir < 0.0) {
	  q = 0.0;
	}

	// Multiply by cell area to get power
        const BoutReal flux =
            q * (pnt.ythis(coord->J) + pnt.ynext(coord->J))
            / (sqrt(pnt.ythis(coord->g_22)) + sqrt(pnt.ynext(coord->g_22)));

        // Divide by volume of cell to get energy loss rate (sign depending on vesheath)
        const BoutReal power = flux / (coord->dy[pnt.ind()] * pnt.ythis(coord->J));

        ASSERT1(std::isfinite(power));
        ASSERT2(power * pnt.dir >= 0.0);

        if (pnt.abs_offset() == 1) {
          energy_source[pnt.ind()] -=
              power * pnt.dir; // Note: Sign negative because power * direction > 0
        }
      }
    }); // end iter_regions

    // Finished boundary conditions for this species
    // Put the modified fields back into the state.
    setBoundary(species["density"], fromFieldAligned(Ni));
    setBoundary(species["temperature"], fromFieldAligned(Ti));
    setBoundary(species["pressure"], fromFieldAligned(Pi));

    if (species.isSet("velocity")) {
      setBoundary(species["velocity"], fromFieldAligned(Vi));
    }

    if (species.isSet("momentum")) {
      setBoundary(species["momentum"], fromFieldAligned(NVi));
    }
    if (tracking) {
      saveParallel(*tracking, fmt::format("NV{}_sheath", kv.first), NVi);
      saveParallel(*tracking, fmt::format("N{}_sheath",kv.first), Ni);
      saveParallel(*tracking, fmt::format("V{}_sheath", kv.first), Vi);
    }
    // Additional loss of energy through sheath
    // Note: Already includes previously set sources
    set(species["energy_source"], fromFieldAligned(energy_source));
  }
  if (tracking) {
    saveParallel(*tracking, "Ne_sheath", Ne);
    if (has_NVe) {
      saveParallel(*tracking, "NVe_sheath", NVe);
    }
    saveParallel(*tracking, "Ve_sheath", Ve);
    saveParallel(*tracking, "phi_sheath", phi);
  }
}
