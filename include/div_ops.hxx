/*
  Finite volume discretisations of divergence operators

  ***********

    Copyright B.Dudson, J.Leddy, University of York, September 2016
              email: benjamin.dudson@york.ac.uk

    This file is part of Hermes.

    Hermes is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Hermes is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Hermes.  If not, see <http://www.gnu.org/licenses/>.

 */

#ifndef DIV_OPS_H
#define DIV_OPS_H

#include <bout/difops.hxx>
#include <bout/field3d.hxx>
#include <bout/fv_ops.hxx>
#include <bout/vector3d.hxx>

/*!
 * Diffusion in index space
 *
 * Similar to using Div_par_diffusion(SQ(mesh->dy)*mesh->g_22, f)
 *
 * @param[in] The field to be differentiated
 * @param[in] bndry_flux  Are fluxes through the boundary calculated?
 */
const Field3D Div_par_diffusion_index(const Field3D& f, bool bndry_flux = true);

const Field3D Div_n_bxGrad_f_B_XPPM(const Field3D& n, const Field3D& f,
                                    bool bndry_flux = true, bool poloidal = false,
                                    bool positive = false);

/// This version has an extra coefficient 'g' that is linearly interpolated
/// onto cell faces
const Field3D Div_n_g_bxGrad_f_B_XZ(const Field3D &n, const Field3D &g, const Field3D &f, 
                                    bool bndry_flux = true, bool positive = false);

const Field3D Div_Perp_Lap_FV_Index(const Field3D& a, const Field3D& f, bool xflux);

const Field3D Div_Z_FV_Index(const Field3D& a, const Field3D& f);

// 4th-order flux conserving term, in index space
const Field3D D4DX4_FV_Index(const Field3D& f, bool bndry_flux = false);
const Field3D D4DZ4_Index(const Field3D& f);

// Div ( k * Grad(f) )
const Field2D Laplace_FV(const Field2D& k, const Field2D& f);

/// Perpendicular diffusion including X and Y directions
/// Takes Div_a_Grad_perp from BOUT++ and adds flows
const Field3D Div_a_Grad_perp_flows(const Field3D& a, const Field3D& f,
                                           Field3D& flux_xlow, Field3D& flux_ylow);
/// Same but with upwinding
/// WARNING: Causes checkerboarding in neutral_mixed integrated test
const Field3D Div_a_Grad_perp_upwind(const Field3D& a, const Field3D& f);

/// Same but with upwinding and flows
/// WARNING: Causes checkerboarding in neutral_mixed integrated test
const Field3D Div_a_Grad_perp_upwind_flows(const Field3D& a, const Field3D& f,
                                           Field3D& flux_xlow, Field3D& flux_ylow);

/// Version with energy flow diagnostic
const Field3D Div_par_K_Grad_par_mod(const Field3D& k, const Field3D& f, Field3D& flow_ylow,
                                     bool bndry_flux = true);

/*!
 * Div ( a Grad_perp(f) ) -- ∇⊥ ( a ⋅ ∇⊥ f) -- Vorticity
 *
 * This version includes corrections for non-orthogonal meshes
 * in which the g12 and g13 components can be non-zero
 * i.e. X-Y, X-Z and Y-Z coordinates can all be non-orthogonal.
 */
const Field3D Div_a_Grad_perp_nonorthog(const Field3D& a, const Field3D& x);

namespace FV {

/// Superbee limiter
///
/// This corresponds to the limiter function
///    φ(r) = max(0, min(2r, 1), min(r,2)
///
/// The value at cell right (i.e. i + 1/2) is:
///
///   n.R = n.c - φ(r) (n.c - (n.p + n.c)/2)
///       = n.c + φ(r) (n.p - n.c)/2
///
/// Four regimes:
///  a) r < 1/2 -> φ(r) = 2r
///     n.R = n.c + gL
///  b) 1/2 < r < 1 -> φ(r) = 1
///     n.R = n.c + gR/2
///  c) 1 < r < 2 -> φ(r) = r
///     n.R = n.c + gL/2
///  d) 2 < r  -> φ(r) = 2
///     n.R = n.c + gR
///
///  where the left and right gradients are:
///   gL = n.c - n.m
///   gR = n.p - n.c
///
struct Superbee {
  void operator()(Stencil1D& n) {
    BoutReal gL = n.c - n.L;
    BoutReal gR = n.R - n.c;

    // r = gL / gR
    // Limiter is φ(r)
    if (gL * gR < 0) {
      // Different signs => Zero gradient
      n.L = n.R = n.c;
    } else {
      BoutReal sign = SIGN(gL);
      gL = fabs(gL);
      gR = fabs(gR);
      BoutReal half_slope = sign * BOUTMAX(BOUTMIN(gL, 0.5 * gR), BOUTMIN(gR, 0.5 * gL));
      n.L = n.c - half_slope;
      n.R = n.c + half_slope;
    }
  }
};

/// This operator calculates Div_par(f v v)
/// It is used primarily (only?) in the parallel momentum equation.
///
/// This operator is used rather than Div(f fv) so that the values of
/// f and v are consistent with other advection equations: The product
/// fv is not interpolated to cell boundaries.
template <typename CellEdges = MC>
const Field3D Div_par_fvv(const Field3D& f_in, const Field3D& v_in,
                          const Field3D& wave_speed_in, bool fixflux = true) {
  ASSERT1_FIELDS_COMPATIBLE(f_in, v_in);
  Mesh* mesh = f_in.getMesh();
  Coordinates* coord = f_in.getCoordinates();
  CellEdges cellboundary;

  if (f_in.isFci()){
    // FCI version, using yup/down fields
    ASSERT1(f_in.hasParallelSlices());
    ASSERT1(v_in.hasParallelSlices());

    Field3D result{emptyFrom(f_in)};

    for (int i = mesh->xstart; i <= mesh->xend; i++) {
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
        for (int k = mesh->zstart; k <= mesh->zend; k++) {
          // Value of f and v at left cell face
          const BoutReal fL = 0.5 * (f_in(i, j, k) + f_in.ydown()(i, j - 1, k));
          const BoutReal vL = 0.5 * (v_in(i, j, k) + v_in.ydown()(i, j - 1, k));

          const BoutReal fR = 0.5 * (f_in(i, j, k) + f_in.yup()(i, j + 1, k));
          const BoutReal vR = 0.5 * (v_in(i, j, k) + v_in.yup()(i, j + 1, k));

          // Reconstruct v at the cell faces
          Stencil1D sv;
          sv.c = v_in(i, j, k);
          sv.m = v_in.ydown()(i, j - 1, k);
          sv.p = v_in.yup()(i, j + 1, k);
          cellboundary(sv);

          // Maximum local wave speed
          const BoutReal amax = BOUTMAX(wave_speed_in
                                        (i, j, k),
                                        fabs(v_in(i, j, k)),
                                        fabs(v_in.yup()(i, j + 1, k)),
                                        fabs(v_in.ydown()(i, j - 1, k)));

          // Calculate flux at right boundary (y+1/2)
          BoutReal fluxRight =
            fR * (vR * vR  + amax * (sv.c - vR)) * (coord->J(i, j, k) + coord->J(i, j + 1, k))
            / (sqrt(coord->g_22(i, j, k)) + sqrt(coord->g_22(i, j + 1, k)));

          // Calculate at left boundary (y-1/2)
          BoutReal fluxLeft =
            fL * (vL * vL - amax * (sv.c - vL)) * (coord->J(i, j, k) + coord->J(i, j - 1, k))
            / (sqrt(coord->g_22(i, j, k)) + sqrt(coord->g_22(i, j - 1, k)));

          result(i, j, k) =
            (fluxRight - fluxLeft) / (coord->dy(i, j, k) * coord->J(i, j, k));
        }
      }
    }
    return result;
  }

  ASSERT1(areFieldsCompatible(f_in, wave_speed_in));

  /// Ensure that f, v and wave_speed are field aligned
  Field3D f = toFieldAligned(f_in, "RGN_NOX");
  Field3D v = toFieldAligned(v_in, "RGN_NOX");
  Field3D wave_speed = toFieldAligned(wave_speed_in, "RGN_NOX");

  Field3D result{zeroFrom(f)};

  // Only need one guard cell, so no need to communicate fluxes
  // Instead calculate in guard cells to preserve fluxes
  int ys = mesh->ystart - 1;
  int ye = mesh->yend + 1;

  for (int i = mesh->xstart; i <= mesh->xend; i++) {

    if (!mesh->firstY(i) || mesh->periodicY(i)) {
      // Calculate in guard cell to get fluxes consistent between processors
      ys = mesh->ystart - 1;
    } else {
      // Don't include the boundary cell. Note that this implies special
      // handling of boundaries later
      ys = mesh->ystart;
    }

    if (!mesh->lastY(i) || mesh->periodicY(i)) {
      // Calculate in guard cells
      ye = mesh->yend + 1;
    } else {
      // Not in boundary cells
      ye = mesh->yend;
    }

    for (int j = ys; j <= ye; j++) {
      // Pre-calculate factors which multiply fluxes


      for (int k = 0; k < mesh->LocalNz; k++) {
	// For right cell boundaries
	BoutReal common_factor = (coord->J(i, j, k) + coord->J(i, j + 1, k))
	  / (sqrt(coord->g_22(i, j, k)) + sqrt(coord->g_22(i, j + 1, k)));
	
	BoutReal flux_factor_rc = common_factor / (coord->dy(i, j, k) * coord->J(i, j, k));
	BoutReal flux_factor_rp = common_factor / (coord->dy(i, j + 1, k) * coord->J(i, j + 1, k));
	
	// For left cell boundaries
	common_factor = (coord->J(i, j, k) + coord->J(i, j - 1, k))
	  / (sqrt(coord->g_22(i, j, k)) + sqrt(coord->g_22(i, j - 1, k)));
	
	BoutReal flux_factor_lc = common_factor / (coord->dy(i, j, k) * coord->J(i, j, k));
	BoutReal flux_factor_lm = common_factor / (coord->dy(i, j - 1, k) * coord->J(i, j - 1, k));

        ////////////////////////////////////////////
        // Reconstruct f at the cell faces
        // This calculates s.R and s.L for the Right and Left
        // face values on this cell

        // Reconstruct f at the cell faces
        Stencil1D s;
        s.c = f(i, j, k);
        s.m = f(i, j - 1, k);
        s.p = f(i, j + 1, k);

        cellboundary(s); // Calculate s.R and s.L

        // Reconstruct v at the cell faces
        Stencil1D sv;
        sv.c = v(i, j, k);
        sv.m = v(i, j - 1, k);
        sv.p = v(i, j + 1, k);

        cellboundary(sv);

        ////////////////////////////////////////////
        // Right boundary

        // Calculate velocity at right boundary (y+1/2)
        BoutReal v_mid = 0.5 * (sv.c + sv.p);
        // And mid-point density at right boundary
        BoutReal n_mid = 0.5 * (s.c + s.p);
        BoutReal flux;

        if (mesh->lastY(i) && (j == mesh->yend) && !mesh->periodicY(i)) {
          // Last point in domain

          if (fixflux) {
            // Use mid-point to be consistent with boundary conditions
            flux = n_mid * v_mid * v_mid;
          } else {
            // Add flux due to difference in boundary values
            flux = s.R * sv.R * sv.R // Use right cell edge values
              + BOUTMAX(wave_speed(i, j, k), fabs(sv.c), fabs(sv.p))
              * n_mid * (sv.R - v_mid); // Damp differences in velocity, not flux
          }
        } else {
          // Maximum wave speed in the two cells
          BoutReal amax = BOUTMAX(wave_speed(i, j, k), wave_speed(i, j + 1, k),
                                  fabs(sv.c), fabs(sv.p));

          flux = s.R * 0.5 * (sv.R + amax) * sv.R;
        }

        result(i, j, k) += flux * flux_factor_rc;
        result(i, j + 1, k) -= flux * flux_factor_rp;

        ////////////////////////////////////////////
        // Calculate at left boundary

        v_mid = 0.5 * (sv.c + sv.m);
        n_mid = 0.5 * (s.c + s.m);

        if (mesh->firstY(i) && (j == mesh->ystart) && !mesh->periodicY(i)) {
          // First point in domain
          if (fixflux) {
            // Use mid-point to be consistent with boundary conditions
            flux = n_mid * v_mid * v_mid;
          } else {
            // Add flux due to difference in boundary values
            flux =
              s.L * sv.L * sv.L
              - BOUTMAX(wave_speed(i, j, k), fabs(sv.c), fabs(sv.m))
              * n_mid * (sv.L - v_mid);
          }
        } else {
          // Maximum wave speed in the two cells
          BoutReal amax = BOUTMAX(wave_speed(i, j, k), wave_speed(i, j - 1, k),
                                  fabs(sv.c), fabs(sv.m));

          flux = s.L * 0.5 * (sv.L - amax) * sv.L;
        }

        result(i, j, k) -= flux * flux_factor_lc;
        result(i, j - 1, k) += flux * flux_factor_lm;
      }
    }
  }
  return fromFieldAligned(result, "RGN_NOBNDRY");
}

// Calculates viscous heating due to numerical momentum fluxes
// and flow of kinetic energy (in flow_ylow)
template <typename CellEdges = MC>
const Field3D Div_par_fvv_heating(const Field3D& f_in, const Field3D& v_in,
                                  const Field3D& wave_speed_in, Field3D &flow_ylow,
                                  bool fixflux = true) {

#if BOUT_USE_METRIC_3D
  throw BoutException("This is currently not ported to FCI yet");
#else

  ASSERT1(areFieldsCompatible(f_in, v_in));
  ASSERT1(areFieldsCompatible(f_in, wave_speed_in));

  Mesh* mesh = f_in.getMesh();

  CellEdges cellboundary;

  /// Ensure that f, v and wave_speed are field aligned
  Field3D f = toFieldAligned(f_in, "RGN_NOX");
  Field3D v = toFieldAligned(v_in, "RGN_NOX");
  Field3D wave_speed = toFieldAligned(wave_speed_in, "RGN_NOX");

  Coordinates* coord = f_in.getCoordinates();

  Field3D result{zeroFrom(f)};
  flow_ylow = zeroFrom(f);

  // Only need one guard cell, so no need to communicate fluxes
  // Instead calculate in guard cells to preserve fluxes
  int ys = mesh->ystart - 1;
  int ye = mesh->yend + 1;

  for (int i = mesh->xstart; i <= mesh->xend; i++) {

    if (!mesh->firstY(i) || mesh->periodicY(i)) {
      // Calculate in guard cell to get fluxes consistent between processors
      ys = mesh->ystart - 1;
    } else {
      // Don't include the boundary cell. Note that this implies special
      // handling of boundaries later
      ys = mesh->ystart;
    }

    if (!mesh->lastY(i) || mesh->periodicY(i)) {
      // Calculate in guard cells
      ye = mesh->yend + 1;
    } else {
      // Not in boundary cells
      ye = mesh->yend;
    }

    for (int j = ys; j <= ye; j++) {
      // Pre-calculate factors which multiply fluxes

      // For right cell boundaries
      BoutReal common_factor = (coord->J(i, j) + coord->J(i, j + 1))
                               / (sqrt(coord->g_22(i, j)) + sqrt(coord->g_22(i, j + 1)));

      BoutReal flux_factor_rc = common_factor / (coord->dy(i, j) * coord->J(i, j));
      BoutReal area_rp = common_factor * coord->dx(i, j + 1) * coord->dz(i, j + 1);

      // For left cell boundaries
      common_factor = (coord->J(i, j) + coord->J(i, j - 1))
                      / (sqrt(coord->g_22(i, j)) + sqrt(coord->g_22(i, j - 1)));

      BoutReal flux_factor_lc = common_factor / (coord->dy(i, j) * coord->J(i, j));
      BoutReal area_lc = common_factor * coord->dx(i, j) * coord->dz(i, j);

      for (int k = 0; k < mesh->LocalNz; k++) {

        ////////////////////////////////////////////
        // Reconstruct f at the cell faces
        // This calculates s.R and s.L for the Right and Left
        // face values on this cell

        // Reconstruct f at the cell faces
        Stencil1D s;
        s.c = f(i, j, k);
        s.m = f(i, j - 1, k);
        s.p = f(i, j + 1, k);

        cellboundary(s); // Calculate s.R and s.L

        // Reconstruct v at the cell faces
        Stencil1D sv;
        sv.c = v(i, j, k);
        sv.m = v(i, j - 1, k);
        sv.p = v(i, j + 1, k);

        cellboundary(sv);

        ////////////////////////////////////////////
        // Right boundary

        // Calculate velocity at right boundary (y+1/2)
        BoutReal v_mid = 0.5 * (sv.c + sv.p);
        // And mid-point density at right boundary
        BoutReal n_mid = 0.5 * (s.c + s.p);

        if (mesh->lastY(i) && (j == mesh->yend) && !mesh->periodicY(i)) {
          // Last point in domain

          // Expected loss of kinetic energy into boundary
          // This is used in the sheath boundary condition to calculate
          // energy losses.
          BoutReal expected_ke = 0.5 * n_mid * v_mid * v_mid * v_mid;

          BoutReal flux_mom;
          if (fixflux) {
            // Mid-point consistent with boundary conditions
            // but kinetic energy loss will not match expected
            // -> Adjust energy balance in pressure equation
            flux_mom = n_mid * v_mid * v_mid;
          } else {
            flux_mom = s.R * sv.R * sv.R
              + BOUTMAX(wave_speed(i, j, k), fabs(sv.c), fabs(sv.p))
              * (s.R * sv.R - n_mid * v_mid);
          }

          // Assume that particle flux is fixed to boundary value
          const BoutReal flux_part = n_mid * v_mid;

          // d/dt(1/2 m n v^2) = v * d/dt(mnv) - 1/2 m v^2 * dn/dt
          BoutReal actual_ke = sv.c * flux_mom - 0.5 * sv.c * sv.c * flux_part;
          
          // Note: If the actual loss was higher than expected, then
          //       plasma heating is needed to compensate
          result(i, j, k) += (actual_ke - expected_ke) * flux_factor_rc;

          // Final flow through boundary is the expected value
          flow_ylow(i, j + 1, k) += expected_ke * area_rp; //expected_ke * area_rp;

        } else {
          // Maximum wave speed in the two cells
          BoutReal amax = BOUTMAX(wave_speed(i, j, k), wave_speed(i, j + 1, k),
                                  fabs(sv.c), fabs(sv.p));

          // Viscous heating due to relaxation of velocity towards midpoint
          result(i, j, k) += (amax + 0.5 * sv.R) * s.R * (sv.c - sv.p) * (sv.R - v_mid) * flux_factor_rc;

          // Kinetic energy flow into next cell.
          // Note: Different from flow out of this cell; the difference
          //       is in the viscous heating.
          BoutReal flux_part = s.R * 0.5 * (sv.R + amax);
          BoutReal flux_mom = flux_part * sv.R;

          flow_ylow(i, j + 1, k) += (sv.p * flux_mom - 0.5 * SQ(sv.p) * flux_part) * area_rp;
        }

        ////////////////////////////////////////////
        // Calculate at left boundary

        v_mid = 0.5 * (sv.c + sv.m);
        n_mid = 0.5 * (s.c + s.m);

        // Expected KE loss. Note minus sign because negative v into boundary
        BoutReal expected_ke = - 0.5 * n_mid * v_mid * v_mid * v_mid;
        
        if (mesh->firstY(i) && (j == mesh->ystart) && !mesh->periodicY(i)) {
          // First point in domain
          BoutReal flux_mom;
          if (fixflux) {
            // Use mid-point to be consistent with boundary conditions
            flux_mom = n_mid * v_mid * v_mid;
          } else {
            // Add flux due to difference in boundary values
            flux_mom =
              s.L * sv.L * sv.L
              - BOUTMAX(wave_speed(i, j, k), fabs(sv.c), fabs(sv.m))
              * (s.L * sv.L - n_mid * v_mid);
          }

          // Assume that density flux is fixed to boundary value
          const BoutReal flux_part = n_mid * v_mid;

          // d/dt(1/2 m n v^2) = v * d/dt(mnv) - 1/2 m v^2 * dn/dt
          BoutReal actual_ke = - sv.c * flux_mom + 0.5 * sv.c * sv.c * flux_part;

          result(i, j, k) += (actual_ke - expected_ke) * flux_factor_lc;

          flow_ylow(i, j, k) -= expected_ke * area_lc;
        } else {
          // Maximum wave speed in the two cells
          BoutReal amax = BOUTMAX(wave_speed(i, j, k), wave_speed(i, j - 1, k),
                                  fabs(sv.c), fabs(sv.m));

          // Viscous heating due to relaxation
          result(i, j, k) += (amax - 0.5 * sv.L) * s.L * (sv.c - sv.m) * (sv.L - v_mid) * flux_factor_lc;

          // Kinetic energy flow into this cell.
          // Note: Different from flow out of left cell; the difference
          //       is in the viscous heating.
          BoutReal flux_part = s.L * 0.5 * (sv.L - amax);
          BoutReal flux_mom = flux_part * sv.L;

          flow_ylow(i, j, k) += (sv.c * flux_mom - 0.5 * SQ(sv.c) * flux_part) * area_lc;
        }
      }
    }
  }
  flow_ylow = fromFieldAligned(flow_ylow, "RGN_NOBNDRY");
  return fromFieldAligned(result, "RGN_NOBNDRY");
#endif
}
  
/// Finite volume parallel divergence
///
/// NOTE: Modified version, applies limiter to velocity and field
///       Performs better (smaller overshoots) than Div_par
///
/// Preserves the sum of f*J*dx*dy*dz over the domain
///
/// @param[in] f_in   The field being advected.
///                   This will be reconstructed at cell faces
///                   using the given CellEdges method
/// @param[in] v_in   The advection velocity.
///                   This will be interpolated to cell boundaries
///                   using linear interpolation
/// @param[in] wave_speed_in  Local maximum speed of all waves in the system at each
//                            point in space
/// @param[in] fixflux     Fix the flux at the boundary to be the value at the
///                        midpoint (for boundary conditions)
///
/// @param[out] flow_ylow    Flow at the lower Y cell boundary
///                          Already includes area factor * flux
///
/// NB: Uses to/from FieldAligned coordinates
template <typename CellEdges = MC>
Field3D Div_par_mod(const Field3D& f_in, const Field3D& v_in,
                          const Field3D& wave_speed_in,
                          Field3D &flow_ylow, bool fixflux = true) {

  if (f_in.isFci()){
    // Use mid-point (cell boundary) averages
    if (flow_ylow.isAllocated()) {
      flow_ylow = emptyFrom(flow_ylow);
    }
    return Div_par(f_in, v_in);
  }
  ASSERT1_FIELDS_COMPATIBLE(f_in, v_in);
  ASSERT1_FIELDS_COMPATIBLE(f_in, wave_speed_in);


  Mesh* mesh = f_in.getMesh();

  CellEdges cellboundary;

  ASSERT2(f_in.getDirectionY() == v_in.getDirectionY());
  ASSERT2(f_in.getDirectionY() == wave_speed_in.getDirectionY());
  const bool are_unaligned =
      ((f_in.getDirectionY() == YDirectionType::Standard)
       and (v_in.getDirectionY() == YDirectionType::Standard)
       and (wave_speed_in.getDirectionY() == YDirectionType::Standard));

  Field3D f = are_unaligned ? toFieldAligned(f_in, "RGN_NOX") : f_in;
  Field3D v = are_unaligned ? toFieldAligned(v_in, "RGN_NOX") : v_in;
  Field3D wave_speed =
      are_unaligned ? toFieldAligned(wave_speed_in, "RGN_NOX") : wave_speed_in;

  Coordinates* coord = f_in.getCoordinates();

  Field3D result{zeroFrom(f)};
  flow_ylow = zeroFrom(f);

  // Only need one guard cell, so no need to communicate fluxes
  // Instead calculate in guard cells to preserve fluxes
  int ys = mesh->ystart - 1;
  int ye = mesh->yend + 1;

  for (int i = mesh->xstart; i <= mesh->xend; i++) {

    if (!mesh->firstY(i) || mesh->periodicY(i)) {
      // Calculate in guard cell to get fluxes consistent between processors
      ys = mesh->ystart - 1;
    } else {
      // Don't include the boundary cell. Note that this implies special
      // handling of boundaries later
      ys = mesh->ystart;
    }

    if (!mesh->lastY(i) || mesh->periodicY(i)) {
      // Calculate in guard cells
      ye = mesh->yend + 1;
    } else {
      // Not in boundary cells
      ye = mesh->yend;
    }

    for (int j = ys; j <= ye; j++) {
      // Pre-calculate factors which multiply fluxes
#if not(BOUT_USE_METRIC_3D)
      // For right cell boundaries
      BoutReal common_factor = (coord->J(i, j) + coord->J(i, j + 1))
                               / (sqrt(coord->g_22(i, j)) + sqrt(coord->g_22(i, j + 1)));

      BoutReal flux_factor_rc = common_factor / (coord->dy(i, j) * coord->J(i, j));
      BoutReal flux_factor_rp =
          common_factor / (coord->dy(i, j + 1) * coord->J(i, j + 1));

      BoutReal area_rp = common_factor * coord->dx(i, j + 1) * coord->dz(i, j + 1);
      
      // For left cell boundaries
      common_factor = (coord->J(i, j) + coord->J(i, j - 1))
                      / (sqrt(coord->g_22(i, j)) + sqrt(coord->g_22(i, j - 1)));

      BoutReal flux_factor_lc = common_factor / (coord->dy(i, j) * coord->J(i, j));
      BoutReal flux_factor_lm =
          common_factor / (coord->dy(i, j - 1) * coord->J(i, j - 1));

      BoutReal area_lc = common_factor * coord->dx(i, j) * coord->dz(i, j);
#endif
      for (int k = 0; k < mesh->LocalNz; k++) {
#if BOUT_USE_METRIC_3D
        // For right cell boundaries
        BoutReal common_factor =
            (coord->J(i, j, k) + coord->J(i, j + 1, k))
            / (sqrt(coord->g_22(i, j, k)) + sqrt(coord->g_22(i, j + 1, k)));
        
        BoutReal flux_factor_rc =
            common_factor / (coord->dy(i, j, k) * coord->J(i, j, k));
        BoutReal flux_factor_rp =
            common_factor / (coord->dy(i, j + 1, k) * coord->J(i, j + 1, k));

        BoutReal area_rp = common_factor * coord->dx(i, j + 1, k) * coord->dz(i, j + 1, k);

        // For left cell boundaries
        common_factor = (coord->J(i, j, k) + coord->J(i, j - 1, k))
                        / (sqrt(coord->g_22(i, j, k)) + sqrt(coord->g_22(i, j - 1, k)));

        BoutReal flux_factor_lc =
            common_factor / (coord->dy(i, j, k) * coord->J(i, j, k));
        BoutReal flux_factor_lm =
            common_factor / (coord->dy(i, j - 1, k) * coord->J(i, j - 1, k));

        BoutReal area_lc = common_factor * coord->dx(i, j, k) * coord->dz(i, j, k);
#endif

        ////////////////////////////////////////////
        // Reconstruct f at the cell faces
        // This calculates s.R and s.L for the Right and Left
        // face values on this cell

        // Reconstruct f at the cell faces
        Stencil1D s;
        s.c = f(i, j, k);
        s.m = f(i, j - 1, k);
        s.p = f(i, j + 1, k);

        cellboundary(s); // Calculate s.R and s.L

        ////////////////////////////////////////////
        // Reconstruct v at the cell faces
        Stencil1D sv;
        sv.c = v(i, j, k);
        sv.m = v(i, j - 1, k);
        sv.p = v(i, j + 1, k);

        cellboundary(sv); // Calculate sv.R and sv.L

        ////////////////////////////////////////////
        // Right boundary

        BoutReal flux;

        if (mesh->lastY(i) && (j == mesh->yend) && !mesh->periodicY(i)) {
          // Last point in domain

          // Calculate velocity at right boundary (y+1/2)
          BoutReal vpar = 0.5 * (v(i, j, k) + v(i, j + 1, k));

          BoutReal bndryval = 0.5 * (s.c + s.p);
          if (fixflux) {
            // Use mid-point to be consistent with boundary conditions
            flux = bndryval * vpar;
          } else {
            // Add flux due to difference in boundary values
            flux = s.R * vpar + wave_speed(i, j, k) * (s.R - bndryval);
          }

        } else {
          // Maximum wave speed in the two cells
          BoutReal amax = BOUTMAX(wave_speed(i, j, k), wave_speed(i, j + 1, k),
                                  fabs(v(i, j, k)), fabs(v(i, j + 1, k)));

          flux = s.R * 0.5 * (sv.R + amax);
        }

        result(i, j, k) += flux * flux_factor_rc;
        result(i, j + 1, k) -= flux * flux_factor_rp;

        flow_ylow(i, j + 1, k) += flux * area_rp;

        ////////////////////////////////////////////
        // Calculate at left boundary

        if (mesh->firstY(i) && (j == mesh->ystart) && !mesh->periodicY(i)) {
          // First point in domain
          BoutReal bndryval = 0.5 * (s.c + s.m);
          BoutReal vpar = 0.5 * (v(i, j, k) + v(i, j - 1, k));
          if (fixflux) {
            // Use mid-point to be consistent with boundary conditions
            flux = bndryval * vpar;
          } else {
            // Add flux due to difference in boundary values
            flux = s.L * vpar - wave_speed(i, j, k) * (s.L - bndryval);
          }
        } else {

          // Maximum wave speed in the two cells
          BoutReal amax = BOUTMAX(wave_speed(i, j, k), wave_speed(i, j - 1, k),
                                  fabs(v(i, j, k)), fabs(v(i, j - 1, k)));

          flux = s.L * 0.5 * (sv.L - amax);
        }

        result(i, j, k) -= flux * flux_factor_lc;
        result(i, j - 1, k) += flux * flux_factor_lm;

        flow_ylow(i, j, k) += flux * area_lc;
      }
    }
  }
  if (are_unaligned) {
    flow_ylow = fromFieldAligned(flow_ylow, "RGN_NOBNDRY");
  }
  return are_unaligned ? fromFieldAligned(result, "RGN_NOBNDRY") : result;
}

/// Div ( a g Grad_perp(f) )  -- Perpendicular gradient-driven advection
///
/// This version uses a slope limiter to calculate cell edge values of g in X,
/// the advects the upwind cell edge.
///
/// 1st order upwinding is used in Y.
template <typename CellEdges = MC>
const Field3D Div_a_Grad_perp_limit(const Field3D& a, const Field3D& g, const Field3D& f) {
#if BOUT_USE_METRIC_3D
  throw BoutException("Currently not supported with FCI");
#else
  ASSERT2(a.getLocation() == f.getLocation());

  Mesh* mesh = a.getMesh();

  // Requires at least 2 communication guard cells in X, 1 in Y
  ASSERT1(mesh->xstart >= 2);
  ASSERT1(mesh->ystart >= 1);

  CellEdges cellboundary;

  Field3D result{zeroFrom(f)};

  Coordinates* coord = f.getCoordinates();

  // Flux in x

  for (int i = mesh->xstart - 1; i <= mesh->xend; i++) {
    for (int j = mesh->ystart; j <= mesh->yend; j++) {
      for (int k = 0; k < mesh->LocalNz; k++) {
        // Calculate flux from i to i+1

        const BoutReal gradient = f(i + 1, j, k) - f(i, j, k);

        // Mid-point average boundary value of 'a'
        const BoutReal aedge = 0.5 * (a(i + 1, j, k) + a(i, j, k));
        BoutReal gedge;
        if (((i == mesh->xstart - 1) and mesh->firstX()) or
            ((i == mesh->xend) and mesh->lastX())) {
          // Mid-point average boundary value of 'g'
          gedge = 0.5 * (g(i + 1, j, k) + g(i, j, k));
        } else if (gradient > 0) {
          // Flux is from (i+1) to (i)
          // Reconstruct `g` at left of (i+1, j, k)

          Stencil1D sg;
          sg.m = g(i, j, k);
          sg.c = g(i + 1, j, k);
          sg.p = g(i + 2, j, k);
          cellboundary(sg); // Calculate sg.R and sg.L

          gedge = sg.L;
        } else {
          // Flux is from (i) to (i+1)
          // Reconstruct `g` at right of (i, j, k)

          Stencil1D sg;
          sg.m = g(i - 1, j, k);
          sg.c = g(i, j, k);
          sg.p = g(i + 1, j, k);
          cellboundary(sg); // Calculate sg.R and sg.L

          gedge = sg.R;
        }

        // Flux across cell edge
        const BoutReal fout = gradient * aedge * gedge
                              * (coord->J(i, j) * coord->g11(i, j)
                                 + coord->J(i + 1, j) * coord->g11(i + 1, j))
                              / (coord->dx(i, j) + coord->dx(i + 1, j));

        result(i, j, k) += fout / (coord->dx(i, j) * coord->J(i, j));
        result(i + 1, j, k) -= fout / (coord->dx(i + 1, j) * coord->J(i + 1, j));
      }
    }
  }
  
  // Y and Z fluxes require Y derivatives

  // Fields containing values along the magnetic field
  Field3D fup(mesh), fdown(mesh);
  Field3D aup(mesh), adown(mesh);
  Field3D gup(mesh), gdown(mesh);

  // Values on this y slice (centre).
  // This is needed because toFieldAligned may modify the field
  Field3D ac = a;
  Field3D gc = g;
  Field3D fc = f;

  // Result of the Y and Z fluxes
  Field3D yzresult(mesh);
  yzresult.allocate();

  if (f.hasParallelSlices() && a.hasParallelSlices() && g.hasParallelSlices()) {
    // All inputs have yup and ydown

    fup = f.yup();
    fdown = f.ydown();

    aup = a.yup();
    adown = a.ydown();

    gup = g.yup();
    gdown = g.ydown();
  } else {
    // At least one input doesn't have yup/ydown fields.
    // Need to shift to/from field aligned coordinates

    fup = fdown = fc = toFieldAligned(f);
    aup = adown = ac = toFieldAligned(a);
    gup = gdown = gc = toFieldAligned(g);
    yzresult.setDirectionY(YDirectionType::Aligned);
  }

  // Y flux

  for (int i = mesh->xstart; i <= mesh->xend; i++) {
    for (int j = mesh->ystart; j <= mesh->yend; j++) {

      BoutReal coef_u =
          0.5
          * (coord->g_23(i, j) / SQ(coord->J(i, j) * coord->Bxy(i, j))
             + coord->g_23(i, j + 1) / SQ(coord->J(i, j + 1) * coord->Bxy(i, j + 1)));

      BoutReal coef_d =
          0.5
          * (coord->g_23(i, j) / SQ(coord->J(i, j) * coord->Bxy(i, j))
             + coord->g_23(i, j - 1) / SQ(coord->J(i, j - 1) * coord->Bxy(i, j - 1)));

      for (int k = 0; k < mesh->LocalNz; k++) {
        // Calculate flux between j and j+1
        int kp = (k + 1) % mesh->LocalNz;
        int km = (k - 1 + mesh->LocalNz) % mesh->LocalNz;

        // Calculate Z derivative at y boundary
        BoutReal dfdz =
            0.25 * (fc(i, j, kp) - fc(i, j, km) + fup(i, j + 1, kp) - fup(i, j + 1, km))
            / coord->dz(i, j);

        // Y derivative
        BoutReal dfdy = 2. * (fup(i, j + 1, k) - fc(i, j, k))
                        / (coord->dy(i, j + 1) + coord->dy(i, j));

        BoutReal aedge = 0.5 * (ac(i, j, k) + aup(i, j + 1, k));
        BoutReal gedge;
        if ((j == mesh->yend) and mesh->lastY(i)) {
          // Midpoint boundary value
          gedge = 0.5 * (gc(i, j, k) + gup(i, j + 1, k));
        } else if (dfdy > 0) {
          // Flux from (j+1) to (j)
          gedge = gup(i, j + 1, k);
        } else {
          // Flux from (j) to (j+1)
          gedge = gc(i, j, k);
        }

        BoutReal fout = aedge * gedge * 0.5
                        * (coord->J(i, j) * coord->g23(i, j)
                           + coord->J(i, j + 1) * coord->g23(i, j + 1))
                        * (dfdz - coef_u * dfdy);

        yzresult(i, j, k) = fout / (coord->dy(i, j) * coord->J(i, j));

        // Calculate flux between j and j-1
        dfdz = 0.25
               * (fc(i, j, kp) - fc(i, j, km) + fdown(i, j - 1, kp) - fdown(i, j - 1, km))
               / coord->dz(i, j);

        dfdy = 2. * (fc(i, j, k) - fdown(i, j - 1, k))
               / (coord->dy(i, j) + coord->dy(i, j - 1));

        aedge = 0.5 * (ac(i, j, k) + adown(i, j - 1, k));
        if ((j == mesh->ystart) and mesh->firstY(i)) {
          gedge = 0.5 * (gc(i, j, k) + gdown(i, j - 1, k));
        } else if (dfdy > 0) {
          gedge = gc(i, j, k);
        } else {
          gedge = gdown(i, j - 1, k);
        }

        fout = aedge * gedge * 0.5
               * (coord->J(i, j) * coord->g23(i, j)
                  + coord->J(i, j - 1) * coord->g23(i, j - 1))
               * (dfdz - coef_d * dfdy);

        yzresult(i, j, k) -= fout / (coord->dy(i, j) * coord->J(i, j));
      }
    }
  }

  // Z flux
  // Easier since all metrics constant in Z

  for (int i = mesh->xstart; i <= mesh->xend; i++) {
    for (int j = mesh->ystart; j <= mesh->yend; j++) {
      // Coefficient in front of df/dy term
      BoutReal coef = coord->g_23(i, j)
                      / (coord->dy(i, j + 1) + 2. * coord->dy(i, j) + coord->dy(i, j - 1))
                      / SQ(coord->J(i, j) * coord->Bxy(i, j));

      for (int k = 0; k < mesh->LocalNz; k++) {
        // Calculate flux between k and k+1
        int kp = (k + 1) % mesh->LocalNz;

        BoutReal gradient =
            // df/dz
            (fc(i, j, kp) - fc(i, j, k)) / coord->dz(i, j)

            // - g_yz * df/dy / SQ(J*B)
            - coef
                  * (fup(i, j + 1, k) + fup(i, j + 1, kp) - fdown(i, j - 1, k)
                     - fdown(i, j - 1, kp));

        BoutReal fout = gradient * 0.5*(ac(i,j,kp) + ac(i,j,k)) *
          ((gradient > 0) ? gc(i, j, kp) : gc(i, j, k));

        yzresult(i, j, k) += fout / coord->dz(i, j);
        yzresult(i, j, kp) -= fout / coord->dz(i, j);
      }
    }
  }
  // Check if we need to transform back
  if (f.hasParallelSlices() && a.hasParallelSlices()) {
    result += yzresult;
  } else {
    result += fromFieldAligned(yzresult);
  }

  return result;
#endif
}

} // namespace FV

namespace FCI {

class dagp_fv {
public:
  Field3D operator()(const Field3D& a, const Field3D& f, Field3D& low_xlow,
                     Field3D& flow_zlow, bool upwinding);
  Field3D operator()(const Field3D& a, const Field3D& f, bool upwinding);
  dagp_fv(Mesh &mesh);
  dagp_fv &operator*=(BoutReal fac) {
    volume /= fac * fac;
    return *this;
  }
  dagp_fv &operator/=(BoutReal fac) { return operator*=(1 / fac); }

private:
  template <bool extra, bool upwinding>
  Field3D operator()(const Field3D& a, const Field3D& f, Field3D* low_xlow,
                     Field3D* flow_zlow);
  Field3D fac_XX;
  Field3D fac_XZ;
  Field3D fac_ZX;
  Field3D fac_ZZ;
  Field3D volume;
  template <bool upwinding>
  BoutReal xflux(const Field3D& a, const Field3D& f, const Ind3D& i);
  template <bool upwinding>
  BoutReal zflux(const Field3D& a, const Field3D& f, const Ind3D& i);
};

std::shared_ptr<dagp_fv>
getDagp_fv(Options& alloptions, Mesh* mesh);
}


#endif //  DIV_OPS_H
