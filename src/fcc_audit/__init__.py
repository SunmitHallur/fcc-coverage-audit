"""FCC 5G Coverage-Change Audit Pipeline.

A re-runnable geospatial pipeline that compares two 6-month vintages of FCC
Broadband Data Collection (BDC) mobile coverage data, infers approximate
cell-site locations, attributes coverage increases to new vs. expanded sites,
and flags provider/county pairs with suspicious growth for FCC manual review.
"""

__version__ = "0.1.0"
