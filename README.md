# Does the scatter of the radial acceleration relation depend on galaxy properties?

Analysis code for PHYS 474 conference paper. Using the SPARC database, the code reproduces the
radial acceleration relation (RAR), then tests whether each galaxy's mean residual about the relation
correlates with surface brightness, gas fraction, size, baryonic mass or morphological type, and
estimates how large a trend the sample could have detected.

## Data

The data are not included. Download them from the SPARC website (http://astroweb.cwru.edu/SPARC/,
Lelli, McGaugh & Schombert 2016):

1. The galaxy table `SPARC_Lelli2016c.mrt`. Save it as `SPARC_Lelli2016c.mrt.txt`.
2. `Rotmod_LTG.zip` (rotation-curve mass models). Unzip it so the `*_rotmod.dat` files are in `Rotmod_LTG/`.

Put both next to `rar_analysis.py`. The column order assumed for the galaxy table is documented in
`read_sparc_table()`; the script stops with an error if it does not match your file.

## Run

```
pip install -r requirements.txt
python rar_analysis.py
```

The full run takes a few minutes (most of the time is the bootstrap and injection-recovery steps).
Figures (PDF) and the point and galaxy tables (CSV) are written to `output/`. The settings (mass-to-light
ratios, sample cuts, number of bootstrap resamples, random seed) are at the top of the script.

## What the script does

1. Computes g_obs = V_obs^2 / R and g_bar from the gas, disk and bulge terms for each radial point
   (fiducial M/L of 0.5 for the disk and 0.7 for the bulge) and fits g_dagger in the McGaugh et al. (2016) function.
2. Fits the intrinsic scatter with a point-level maximum-likelihood model that includes velocity,
   inclination, distance and M/L errors (errors are treated as independent between points).
3. Tests the per-galaxy mean residual against five properties with Spearman correlations and
   weighted slopes, using galaxy-level bootstrap intervals and Benjamini-Hochberg correction.
4. Injection-recovery: finds the smallest trend detected in 80% of trials.
5. Exploratory checks: low versus high acceleration galaxies, a high versus low surface brightness
   KS test, and the residual against normalised radius.
6. Robustness: other M/L ratios and a Q = 1 subsample.

## Reproducibility

A single random generator (seed 474) is used for all resampling, so results are repeatable. Bootstrap
intervals can differ in the last digit if the order of the steps is changed.

## Reference

Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). SPARC: Mass models for 175 disk galaxies with
Spitzer photometry and accurate rotation curves. The Astronomical Journal, 152, 157.
