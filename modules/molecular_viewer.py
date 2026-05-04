"""
molecular_viewer.py
3D interactive visualization for catalyst surface slabs and metabolite structures.
Uses 3Dmol.js (CDN) via st.components.v1.html — no extra Python packages required.
"""

import uuid
import random
import numpy as np

# ─── Atomic radii (pm) for surface slab geometry ─────────────────────────────
_RADIUS_PM: dict[str, float] = {
    "Cu": 128, "Pt": 136, "Pd": 137, "Fe": 126, "Ni": 124, "Co": 125,
    "Ru": 134, "Rh": 134, "Au": 144, "Ag": 144, "Ir": 136, "Mo": 139,
    "W":  139, "Zn": 122, "In": 167, "Sn": 141, "Al": 143, "Ti": 147,
    "Mn": 127, "Ga": 122, "Ce": 182, "Zr": 155, "C":  77,  "Si": 111,
}

# ─── Metabolite XYZ library (Angstrom) ───────────────────────────────────────
# Small molecules: full atoms (incl. H).  Larger molecules: heavy atoms only.

MOLECULE_XYZ: dict[str, str] = {

    "CO2": """\
3
Carbon Dioxide
C   0.000   0.000   0.000
O   1.163   0.000   0.000
O  -1.163   0.000   0.000""",

    "CO": """\
2
Carbon Monoxide
C   0.000   0.000   0.000
O   1.128   0.000   0.000""",

    "H2": """\
2
Dihydrogen
H   0.000   0.000   0.000
H   0.741   0.000   0.000""",

    "N2": """\
2
Dinitrogen
N   0.000   0.000   0.000
N   1.098   0.000   0.000""",

    "H2O": """\
3
Water
O   0.000   0.000   0.117
H   0.000   0.757  -0.469
H   0.000  -0.757  -0.469""",

    "NH3": """\
4
Ammonia
N   0.000   0.000   0.116
H   0.000   0.940  -0.271
H   0.814  -0.470  -0.271
H  -0.814  -0.470  -0.271""",

    "CH4": """\
5
Methane
C   0.000   0.000   0.000
H   0.628   0.628   0.628
H  -0.628  -0.628   0.628
H  -0.628   0.628  -0.628
H   0.628  -0.628  -0.628""",

    "Methanol": """\
6
Methanol
C   0.000   0.000   0.000
O   1.430   0.000   0.000
H  -0.393   1.028   0.000
H  -0.393  -0.514   0.890
H  -0.393  -0.514  -0.890
H   1.113   0.907   0.000""",

    "Ethanol": """\
9
Ethanol
C   0.000   0.000   0.000
C   1.540   0.000   0.000
O   2.098   1.267   0.000
H  -0.393   1.028   0.000
H  -0.393  -0.514   0.890
H  -0.393  -0.514  -0.890
H   1.930  -0.514   0.890
H   1.930  -0.514  -0.890
H   1.630   1.857   0.000""",

    "Acetaldehyde": """\
7
Acetaldehyde
C   0.000   0.000   0.000
C   1.520   0.000   0.000
O   2.128   1.060   0.000
H  -0.393   1.028   0.000
H  -0.393  -0.514   0.890
H  -0.393  -0.514  -0.890
H   2.080  -0.950   0.000""",

    "Pyruvate": """\
6
Pyruvate (heavy atoms)
C   0.000   0.000   0.000
C   1.520   0.000   0.000
O   2.140   1.060   0.000
C   2.300  -1.200   0.000
O   1.600  -2.250   0.000
O   3.520  -1.200   0.000""",

    "Lactic Acid": """\
6
Lactic Acid (heavy atoms)
C   0.000   0.000   0.000
C   1.540   0.000   0.000
O   2.080   1.280   0.000
C   2.300  -1.200   0.000
O   1.580  -2.260   0.000
O   3.520  -1.180   0.000""",

    "Lactate": """\
6
Lactate (heavy atoms)
C   0.000   0.000   0.000
C   1.540   0.000   0.000
O   2.080   1.280   0.000
C   2.300  -1.200   0.000
O   1.580  -2.260   0.000
O   3.520  -1.180   0.000""",

    "Succinate": """\
8
Succinate (heavy atoms)
C   0.000   0.000   0.000
C   1.540   0.000   0.000
C   2.100   1.400   0.000
C  -0.560   1.400   0.000
O  -1.790   1.490   0.000
O   0.100   2.520   0.000
O   3.340   1.500   0.000
O   1.490   2.540   0.000""",

    "Succinic Acid": """\
8
Succinic Acid (heavy atoms)
C   0.000   0.000   0.000
C   1.540   0.000   0.000
C   2.100   1.400   0.000
C  -0.560   1.400   0.000
O  -1.790   1.490   0.000
O   0.100   2.520   0.000
O   3.340   1.500   0.000
O   1.490   2.540   0.000""",

    "Oxaloacetate": """\
7
Oxaloacetate (heavy atoms)
C   0.000   0.000   0.000
C   1.520   0.000   0.000
O   2.140   1.060   0.000
C   2.300  -1.200   0.000
O   1.600  -2.250   0.000
O   3.520  -1.200   0.000
O  -1.230   0.000   0.000""",

    "Malate": """\
7
Malate (heavy atoms)
C   0.000   0.000   0.000
C   1.540   0.000   0.000
O   2.080   1.280   0.000
C   2.300  -1.200   0.000
O   1.600  -2.250   0.000
O   3.520  -1.200   0.000
O  -1.230   0.000   0.000""",

    "Fumarate": """\
6
Fumarate (heavy atoms)
C   0.000   0.000   0.000
C   1.340   0.000   0.000
C  -0.710  -1.230   0.000
C   2.050  -1.230   0.000
O  -0.200  -2.360   0.000
O   1.540  -2.360   0.000""",

    "Glucose": """\
12
Glucose — β-D-glucopyranose ring (heavy atoms)
O   0.000   0.000   0.000
C  -0.705   1.220   0.000
C   0.705   2.045   0.512
C   1.415   1.220   1.200
C   0.705  -0.025   1.200
C  -0.705  -0.845   1.200
O   1.295   3.445   0.090
O   2.820   1.060   1.200
O   1.305  -1.145   1.200
O  -0.705  -2.270   1.200
O  -2.145   0.000   0.000
C  -2.100  -1.220   1.200""",

    "Xylose": """\
10
Xylose (heavy atoms)
O   0.000   0.000   0.000
C  -0.705   1.220   0.000
C   0.705   2.045   0.512
C   1.415   1.220   1.200
C   0.705  -0.025   1.200
O   1.295   3.445   0.090
O   2.820   1.060   1.200
O   1.305  -1.145   1.200
O  -2.145   0.000   0.000
C  -0.705  -0.845   1.200""",

    "Isobutanol": """\
5
Isobutanol (heavy atoms)
C   0.000   0.000   0.000
C   1.540   0.000   0.000
O   2.080   1.280   0.000
C   2.230  -1.100   0.000
C   2.230   0.000   1.540""",

    "Isobutyraldehyde": """\
5
Isobutyraldehyde (heavy atoms)
C   0.000   0.000   0.000
C   1.520   0.000   0.000
O   2.140   1.060   0.000
C  -0.770   1.270   0.000
C  -0.770  -1.270   0.000""",

    "2-Ketoisovalerate": """\
7
2-Ketoisovalerate (heavy atoms)
C   0.000   0.000   0.000
C   1.520   0.000   0.000
O   2.140   1.060   0.000
C   2.300  -1.200   0.000
O   1.600  -2.250   0.000
O   3.520  -1.200   0.000
C  -1.540   0.000   0.000""",

    "1,3-Propanediol": """\
5
1,3-Propanediol (heavy atoms)
O  -1.430   0.000   0.000
C   0.000   0.000   0.000
C   1.540   0.000   0.000
C   2.090   1.400   0.000
O   3.510   1.390   0.000""",

    "3-Hydroxypropionaldehyde": """\
5
3-Hydroxypropionaldehyde (heavy atoms)
O  -1.230   0.000   0.000
C   0.000   0.000   0.000
C   1.540   0.000   0.000
C   2.090   1.400   0.000
O   2.730   1.400   1.060""",

    "Glycerol": """\
6
Glycerol (heavy atoms)
O  -1.430   0.000   0.000
C   0.000   0.000   0.000
C   1.540   0.000   0.000
O   2.100   1.270   0.000
C   2.090  -1.400   0.000
O   3.510  -1.390   0.000""",

    "Fatty Acid": """\
14
Palmitic Acid C16 (heavy atoms)
C   0.000   0.000   0.000
C   1.540   0.000   0.000
C   3.080   0.000   0.000
C   4.620   0.000   0.000
C   6.160   0.000   0.000
C   7.700   0.000   0.000
C   9.240   0.000   0.000
C  10.780   0.000   0.000
C  12.320   0.000   0.000
C  13.860   0.000   0.000
C  15.400   0.000   0.000
C  16.940   0.000   0.000
C  18.480   0.000   0.000
O  19.100   1.060   0.000""",

    "Farnesene": """\
15
beta-Farnesene C15H24 (heavy atoms)
C   0.000   0.000   0.000
C   1.340   0.000   0.000
C   2.100   1.230   0.000
C   3.580   1.230   0.000
C   4.350   0.000   0.000
C   4.350   2.460   0.000
C   5.890   0.000   0.000
C   6.660   1.230   0.000
C   8.140   1.230   0.000
C   8.910   0.000   0.000
C   8.910   2.460   0.000
C  10.450   0.000   0.000
C  11.220   1.230   0.000
C  12.700   1.230   0.000
C  13.470   0.000   0.000""",

    "Acetyl-CoA": """\
7
Acetyl-CoA (thioester region, heavy atoms)
C   0.000   0.000   0.000
C   1.520   0.000   0.000
O   2.140   1.060   0.000
S   2.300  -1.450   0.000
C   3.850  -1.450   0.000
C   5.390  -1.450   0.000
N   6.000  -0.200   0.000""",

    "Acetoacetyl-CoA": """\
9
Acetoacetyl-CoA (key atoms)
C   0.000   0.000   0.000
C   1.520   0.000   0.000
O   2.140   1.060   0.000
C   2.300  -1.200   0.000
O   1.600  -2.250   0.000
C   3.840  -1.200   0.000
O   4.460  -0.140   0.000
S   4.460  -2.460   0.000
C   6.000  -2.460   0.000""",

    "Mevalonate": """\
7
Mevalonate (heavy atoms)
O   0.000   0.000   0.000
C   1.430   0.000   0.000
C   2.970   0.000   0.000
O   3.510   1.270   0.000
C   3.510  -1.270   0.000
C   4.960  -1.270   0.000
O   5.500   0.000   0.000""",

    "FPP": """\
15
Farnesyl Pyrophosphate (heavy atoms, simplified)
C   0.000   0.000   0.000
C   1.340   0.000   0.000
C   2.100   1.230   0.000
C   3.580   1.230   0.000
C   4.350   0.000   0.000
C   5.890   0.000   0.000
C   6.660   1.230   0.000
C   8.140   1.230   0.000
C   8.910   0.000   0.000
C  10.450   0.000   0.000
C  11.220   1.230   0.000
C  12.700   1.230   0.000
O  13.430   2.360   0.000
P  14.890   2.360   0.000
P  16.350   2.360   0.000""",

    "IPP": """\
5
Isopentenyl Pyrophosphate (heavy atoms, simplified)
C   0.000   0.000   0.000
C   1.340   0.000   0.000
C   2.100   1.230   0.000
O   3.430   1.230   0.000
P   4.970   1.230   0.000""",

    "Bicarbonate": """\
4
Bicarbonate
C   0.000   0.000   0.000
O   1.163   0.000   0.000
O  -0.580   1.005   0.000
O  -0.580  -1.005   0.000""",

    "Phosphoenolpyruvate": """\
8
Phosphoenolpyruvate (heavy atoms)
C   0.000   0.000   0.000
C   1.340   0.000   0.000
C   2.100   1.230   0.000
O   1.430   2.380   0.000
O   3.330   1.200   0.000
P   4.870   1.200   0.000
C   2.840  -1.060   0.000
O   2.200  -2.120   0.000""",

    "Malonyl-CoA": """\
7
Malonyl-CoA (simplified heavy atoms)
C   0.000   0.000   0.000
O  -0.620   1.060   0.000
O  -0.620  -1.060   0.000
C   1.540   0.000   0.000
O   2.160   1.060   0.000
S   2.300  -1.450   0.000
C   3.850  -1.450   0.000""",

    "Fatty Acyl-CoA": """\
12
Fatty Acyl-CoA (simplified heavy atoms)
C   0.000   0.000   0.000
C   1.540   0.000   0.000
C   3.080   0.000   0.000
C   4.620   0.000   0.000
C   6.160   0.000   0.000
C   7.700   0.000   0.000
C   9.240   0.000   0.000
C  10.780   0.000   0.000
O  11.400   1.060   0.000
S  11.400  -1.450   0.000
C  12.940  -1.450   0.000
N  13.560  -0.200   0.000""",

    "PHB": """\
8
Polyhydroxybutyrate repeat unit
C   0.000   0.000   0.000
C   1.540   0.000   0.000
O   2.080   1.280   0.000
C   2.300  -1.200   0.000
O   1.600  -2.250   0.000
C   3.840  -1.200   0.000
C   5.380  -1.200   0.000
O   5.920  -2.460   0.000""",

    "DHAP": """\
6
Dihydroxyacetone Phosphate (heavy atoms)
O   0.000   0.000   0.000
C   1.230   0.000   0.000
C   2.460   0.000   0.000
O   2.460   1.430   0.000
C   3.690   0.000   0.000
P   5.230   0.000   0.000""",

    "FAEE": """\
14
Fatty Acid Ethyl Ester (heavy atoms)
C   0.000   0.000   0.000
C   1.430   0.000   0.000
O   2.050   1.270   0.000
C   3.480   1.270   0.000
C   5.020   1.270   0.000
C   6.560   1.270   0.000
C   8.100   1.270   0.000
C   9.640   1.270   0.000
C  11.180   1.270   0.000
C  12.720   1.270   0.000
C  14.260   1.270   0.000
C  15.800   1.270   0.000
C  17.340   1.270   0.000
O  17.960   0.210   0.000""",

    # Phosphorylated metabolites — show phosphate group prominently
    "Glucose-6-P": """\
13
Glucose-6-Phosphate (heavy atoms)
O   0.000   0.000   0.000
C  -0.705   1.220   0.000
C   0.705   2.045   0.512
C   1.415   1.220   1.200
C   0.705  -0.025   1.200
C  -0.705  -0.845   1.200
O   1.295   3.445   0.090
O   2.820   1.060   1.200
O   1.305  -1.145   1.200
O  -2.145   0.000   0.000
C  -2.100  -1.220   1.200
O  -3.540  -1.220   1.200
P  -5.080  -1.220   1.200""",

    "Fructose-6-P": """\
13
Fructose-6-Phosphate (heavy atoms)
O   0.000   0.000   0.000
C  -0.705   1.220   0.000
C   0.705   2.045   0.512
C   1.415   1.220   1.200
C   0.705  -0.025   1.200
C  -0.705  -0.845   1.200
O   0.705   3.520   0.512
O   2.820   1.060   1.200
O   1.305  -1.145   1.200
O  -2.145   0.000   0.000
C  -2.100  -1.220   1.200
O  -3.540  -1.220   1.200
P  -5.080  -1.220   1.200""",

    "Fructose-1,6-BP": """\
14
Fructose-1,6-Bisphosphate (heavy atoms)
P  -6.620  -1.220   1.200
O  -5.080  -1.220   1.200
C  -3.540  -1.220   1.200
C  -2.100  -1.220   1.200
O  -2.145   0.000   0.000
C  -0.705   1.220   0.000
C   0.705   2.045   0.512
C   1.415   1.220   1.200
C   0.705  -0.025   1.200
O   2.820   1.060   1.200
O   1.305  -1.145   1.200
O   0.705   3.520   0.512
O  -0.705  -0.845   1.200
P   2.960  -1.145   1.200""",

    "Xylulose": """\
10
Xylulose (heavy atoms)
O   0.000   0.000   0.000
C   1.220   0.000   0.000
C   1.840   1.280   0.000
C   1.840  -1.280   0.000
O   0.610  -1.280   0.000
C   3.280   1.280   0.000
O   3.900   0.210   0.000
O   3.280   2.560   0.000
C   3.280  -1.280   0.000
O   3.900  -2.350   0.000""",

    "Xylulose-5-P": """\
11
Xylulose-5-Phosphate (heavy atoms)
P   0.000   0.000   0.000
O   1.540   0.000   0.000
C   2.160   1.270   0.000
C   3.700   1.270   0.000
O   4.320   0.210   0.000
C   4.320   2.540   0.000
O   3.700   3.610   0.000
C   5.860   2.540   0.000
O   6.480   1.470   0.000
C   5.860   3.810   0.000
O   5.240   4.880   0.000""",

    "Glycerol-3-P": """\
7
Glycerol-3-Phosphate (heavy atoms)
P   0.000   0.000   0.000
O   1.540   0.000   0.000
C   2.160   1.270   0.000
C   3.700   1.270   0.000
O   4.320   2.540   0.000
C   4.320   0.000   0.000
O   5.760   0.000   0.000""",

    "HMG-CoA": """\
10
HMG-CoA (simplified heavy atoms)
C   0.000   0.000   0.000
C   1.520   0.000   0.000
O   2.140   1.060   0.000
C   2.300  -1.200   0.000
O   1.600  -2.250   0.000
C   3.840  -1.200   0.000
C   4.460   0.000   0.000
O   4.460   1.430   0.000
S   5.990   0.000   0.000
C   7.530   0.000   0.000""",

    "Acetolactate": """\
7
Acetolactate (heavy atoms)
C   0.000   0.000   0.000
C   1.520   0.000   0.000
O   2.140   1.060   0.000
C   2.300  -1.200   0.000
O   1.600  -2.250   0.000
O   3.520  -1.200   0.000
C   3.840   0.000   0.000""",

    "2,3-Dihydroxyisovalerate": """\
7
2,3-Dihydroxyisovalerate (heavy atoms)
C   0.000   0.000   0.000
C   1.540   0.000   0.000
O   2.080   1.280   0.000
C   2.300  -1.200   0.000
O   1.600  -2.250   0.000
O   3.520  -1.200   0.000
C  -1.540   0.000   0.000""",
}

# ─── Pathway node label → MOLECULE_XYZ key ───────────────────────────────────

_NODE_MAP: dict[str, str] = {
    # Gases / inorganics
    "co₂":                          "CO2",
    "co2":                          "CO2",
    "co₂ + h₂o":                   "CO2",
    "h₂":                           "H2",
    "h2":                           "H2",
    "n₂":                           "N2",
    "n2":                           "N2",
    "co":                           "CO",
    "h₂o":                          "H2O",
    "h2o":                          "H2O",
    "water":                        "H2O",
    "nh₃":                          "NH3",
    "nh3":                          "NH3",
    "ammonia":                      "NH3",
    "ch₄":                          "CH4",
    "ch4":                          "CH4",
    "methane":                      "CH4",
    "bicarbonate":                  "Bicarbonate",
    # C1 / simple alcohols
    "methanol":                     "Methanol",
    "ethanol":                      "Ethanol",
    "fatty acids + ethanol":        "Ethanol",
    "acetaldehyde":                 "Acetaldehyde",
    # Sugars
    "glucose":                      "Glucose",
    "xylose":                       "Xylose",
    "xylose / glucose":             "Xylose",
    "xylulose":                     "Xylulose",
    "xylulose-5-p":                 "Xylulose-5-P",
    "fructose-6-p":                 "Fructose-6-P",
    "fructose-1,6-bp":              "Fructose-1,6-BP",
    "glucose-6-p":                  "Glucose-6-P",
    "glucose 6-phosphate":          "Glucose-6-P",
    "ribulose 5-phosphate":         "Xylulose-5-P",
    "dhap":                         "DHAP",
    "glycerol":                     "Glycerol",
    "glycerol-3-p":                 "Glycerol-3-P",
    # Organic acids / TCA
    "pyruvate":                     "Pyruvate",
    "pyruvate (via ppp)":           "Pyruvate",
    "oxaloacetate":                 "Oxaloacetate",
    "malate":                       "Malate",
    "fumarate":                     "Fumarate",
    "succinate":                    "Succinate",
    "succinic acid":                "Succinic Acid",
    "lactate":                      "Lactate",
    "lactic acid":                  "Lactic Acid",
    "phosphoenolpyruvate":          "Phosphoenolpyruvate",
    # Isobutanol pathway
    "acetolactate":                 "Acetolactate",
    "2,3-dihydroxyisovalerate":     "2,3-Dihydroxyisovalerate",
    "2-ketoisovalerate":            "2-Ketoisovalerate",
    "isobutyraldehyde":             "Isobutyraldehyde",
    "isobutanol":                   "Isobutanol",
    # Mevalonate / terpenoid
    "acetyl-coa":                   "Acetyl-CoA",
    "acetoacetyl-coa":              "Acetoacetyl-CoA",
    "hmg-coa":                      "HMG-CoA",
    "mevalonate":                   "Mevalonate",
    "ipp/dmapp":                    "IPP",
    "fpp":                          "FPP",
    "farnesene":                    "Farnesene",
    "β-farnesene (c₁₅h₂₄)":        "Farnesene",
    # Fatty acid / lipid
    "malonyl-coa":                  "Malonyl-CoA",
    "fatty acyl-acp":               "Fatty Acyl-CoA",
    "fatty acyl-coa":               "Fatty Acyl-CoA",
    "fatty acyl-coa + ethanol":     "Fatty Acyl-CoA",
    "fatty acid":                   "Fatty Acid",
    "fatty acids":                  "Fatty Acid",
    "faee":                         "FAEE",
    "fatty acid ethyl esters (faee)":"FAEE",
    # Bioplastic
    "(r)-3-hydroxybutyryl-coa":     "Acetoacetyl-CoA",
    "polyhydroxybutyrate (phb)":    "PHB",
    "phb polymer":                  "PHB",
    # PDO
    "3-hydroxypropionaldehyde":     "3-Hydroxypropionaldehyde",
    "1,3-propanediol":              "1,3-Propanediol",
}


def _lookup_mol(node_label: str) -> tuple[str, str]:
    """Return (xyz_str, display_name) for a pathway node label.
    Falls back to a simple C-chain stub if the node is unknown.
    """
    key = _NODE_MAP.get(node_label.lower().strip())
    if key and key in MOLECULE_XYZ:
        return MOLECULE_XYZ[key], key

    # Fuzzy: try to find a substring match
    low = node_label.lower()
    for map_key, mol_key in _NODE_MAP.items():
        if map_key in low or low in map_key:
            if mol_key in MOLECULE_XYZ:
                return MOLECULE_XYZ[mol_key], mol_key

    # Ultimate fallback: generic 4-carbon chain
    fallback_xyz = "4\nGeneric metabolite\nC 0 0 0\nC 1.54 0 0\nC 3.08 0 0\nO 3.70 1.06 0"
    return fallback_xyz, node_label


def get_molecule_xyz(node_label: str) -> tuple[str, str]:
    """Public API: return (xyz_str, mol_name) for a given pathway node label."""
    return _lookup_mol(node_label)


# ─── Catalyst surface slab generator ─────────────────────────────────────────

def _build_element_pool(composition: dict, seed: int = 42) -> list[str]:
    total = sum(composition.values()) or 1.0
    pool: list[str] = []
    for el, frac in composition.items():
        pool.extend([el] * max(1, round(frac / total * 100)))
    return pool


def generate_surface_xyz(catalyst: dict) -> str:
    """Generate an XYZ string for a catalyst surface slab (FCC geometry)."""
    comp   = catalyst.get("composition", {"Cu": 1.0})
    facet  = catalyst.get("surface_facet", "(111)")
    total  = sum(comp.values()) or 1.0
    norm   = {e: v / total for e, v in comp.items()}

    # Weighted-average atomic radius → nearest-neighbour distance (Å)
    avg_r_pm = sum(_RADIUS_PM.get(e, 130) * f for e, f in norm.items())
    d_nn = avg_r_pm * 2.0 * 0.01           # FCC: d_nn = 2r in Å

    pool = _build_element_pool(norm)
    rng  = random.Random(42)
    atoms: list[tuple[str, float, float, float]] = []

    n_x, n_y, n_layers = 5, 5, 4

    if "(100)" in facet:
        d_layer = d_nn / 1.414
        for k in range(n_layers):
            off = d_nn / 2 if k % 2 else 0.0
            for i in range(n_x):
                for j in range(n_y):
                    atoms.append((rng.choice(pool),
                                  i * d_nn + off,
                                  j * d_nn + off,
                                  k * d_layer))
    elif "(110)" in facet:
        d_row   = d_nn
        d_col   = d_nn * 1.414
        d_layer = d_nn / 1.414
        for k in range(n_layers):
            off = d_nn / 2 if k % 2 else 0.0
            for i in range(n_x):
                for j in range(3):
                    atoms.append((rng.choice(pool),
                                  i * d_row + off,
                                  j * d_col,
                                  k * d_layer))
    else:
        # Default: FCC (111) — hexagonal close-packed layers, ABC stacking
        d_layer = d_nn * np.sqrt(2.0 / 3.0)
        layer_offsets = [
            (0.0,              0.0),
            (d_nn / 2.0,       d_nn / (2.0 * np.sqrt(3.0))),
            (0.0,              d_nn / np.sqrt(3.0)),
        ]
        for k in range(n_layers):
            ox, oy = layer_offsets[k % 3]
            for i in range(n_x):
                for j in range(n_y):
                    x = i * d_nn + j * d_nn / 2.0 + ox
                    y = j * d_nn * np.sqrt(3.0) / 2.0 + oy
                    z = k * d_layer
                    atoms.append((rng.choice(pool), x, y, z))

    lines = [str(len(atoms)), f"Catalyst surface: {catalyst.get('name', '')}"]
    for el, x, y, z in atoms:
        lines.append(f"{el}  {x:.4f}  {y:.4f}  {z:.4f}")
    return "\n".join(lines)


# ─── 3Dmol.js HTML renderer ───────────────────────────────────────────────────

_3DMOL_CDN = (
    "https://cdnjs.cloudflare.com/ajax/libs/"
    "3Dmol/2.0.4/3Dmol-min.js"
)


def make_viewer_html(
    xyz: str,
    *,
    width: int    = 680,
    height: int   = 420,
    spin: bool    = True,
    style: str    = "ballstick",   # "ballstick" | "sphere" | "stick"
    bg_color: str = "0x0e1117",
    label: str    = "",
) -> str:
    """Return a self-contained HTML string for a 3Dmol.js interactive viewer."""

    uid = uuid.uuid4().hex[:8]

    # JavaScript style spec
    if style == "sphere":
        style_js = '{"sphere":{"scale":0.45,"colorscheme":"Jmol"}}'
    elif style == "stick":
        style_js = '{"stick":{"radius":0.18,"colorscheme":"Jmol"}}'
    else:
        style_js = (
            '{"sphere":{"scale":0.32,"colorscheme":"Jmol"},'
            '"stick":{"radius":0.12,"colorscheme":"Jmol"}}'
        )

    spin_js = 'v.spin("y", 0.5);' if spin else ""

    # Escape for JS template literal
    xyz_safe  = xyz.replace("\\", "\\\\").replace("`", "\\`")
    label_tag = (
        f'<div style="color:#86868b;font:11px/1 Inter,sans-serif;'
        f'text-align:center;padding:5px 0 0;">{label}</div>'
        if label else ""
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<script src="{_3DMOL_CDN}"></script>
<style>
  body{{margin:0;padding:0;background:{bg_color.replace("0x","#")};}}
  #v{uid}{{width:{width}px;height:{height}px;position:relative;}}
</style>
</head>
<body>
<div id="v{uid}"></div>
{label_tag}
<script>
  let v = $3Dmol.createViewer(document.getElementById("v{uid}"),
                               {{backgroundColor:"{bg_color}"}});
  v.addModel(`{xyz_safe}`, "xyz");
  v.setStyle({{}}, {style_js});
  v.zoomTo();
  {spin_js}
  v.render();
</script>
</body>
</html>"""


def make_surface_viewer_html(catalyst: dict, **kwargs) -> str:
    """Convenience wrapper: generate surface slab XYZ then render HTML."""
    xyz = generate_surface_xyz(catalyst)
    label = (
        f"{catalyst.get('name','')} · {catalyst.get('surface_facet','(111)')} "
        f"· {len(catalyst.get('composition',{}))} elements"
    )
    return make_viewer_html(
        xyz,
        style=kwargs.get("style", "sphere"),
        spin=kwargs.get("spin", True),
        label=label,
        width=kwargs.get("width", 680),
        height=kwargs.get("height", 420),
    )


def make_molecule_viewer_html(node_label: str, **kwargs) -> tuple[str, str]:
    """Return (html_str, mol_name) for a pathway node label."""
    xyz, mol_name = get_molecule_xyz(node_label)
    html = make_viewer_html(
        xyz,
        style=kwargs.get("style", "ballstick"),
        spin=kwargs.get("spin", True),
        label=mol_name,
        width=kwargs.get("width", 680),
        height=kwargs.get("height", 400),
    )
    return html, mol_name
