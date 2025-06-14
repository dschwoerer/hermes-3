#pragma once
#ifndef DIAMAGNETIC_DRIFT_H
#define DIAMAGNETIC_DRIFT_H

#include <bout/vectormetric.hxx>
#include <bout/yboundary_regions.hxx>

#include "component.hxx"

/// Calculate diamagnetic flows

struct DiamagneticDrift : public Component {
  DiamagneticDrift(std::string name, Options &options, Solver *UNUSED(solver));

  /// For every species, if it has:
  ///  - temperature
  ///  - charge
  ///
  /// Modifies:
  ///  - density_source
  ///  - energy_source
  ///  - momentum_source
  void transform(Options &state) override;

private:
  VectorMetric Curlb_B;
  bool bndry_flux;
  Coordinates::FieldMetric diamag_form;
  YBoundary yboundary;
};

namespace {
RegisterComponent<DiamagneticDrift> registercomponentdiamagnetic("diamagnetic_drift");
}

#endif // DIAMAGNETIC_DRIFT_H


