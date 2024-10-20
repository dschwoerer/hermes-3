#pragma once
#ifndef CLASSICAL_DIFFUSION_H
#define CLASSICAL_DIFFUSION_H

#include "component.hxx"
#include <bout/vectormetric.hxx>

struct ClassicalDiffusion : public Component {
  ClassicalDiffusion(std::string name, Options& alloptions, Solver*);

  void transform(Options &state) override;

  void outputVars(Options &state) override;
private:
  Field3D Bsq; // Magnetic field squared

  bool diagnose; ///< Output additional diagnostics?
  Field3D Dn; ///< Particle diffusion coefficient
  BoutReal custom_D; ///< User-set particle diffusion coefficient override
};

namespace {
RegisterComponent<ClassicalDiffusion> registercomponentclassicaldiffusion("classical_diffusion");
}

#endif // CLASSICAL_DIFFUSION_H
