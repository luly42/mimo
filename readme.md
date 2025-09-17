**README.md**

**MiMO: Mixture Model for Open Clusters in Color–Magnitude Diagrams**

**Overview**

MiMO is a powerful tool designed to extract essential properties of open clusters (OCs) from color-magnitude diagrams (CMDs). Including isochrone parameters (age, distance, metallicity, and dust extinction), stellar mass function, and binary parameters (binary fraction and mass ratio distribution)

We treats an OC in the CMD as a mixture of single and binary member stars, and field stars in the same region. The cluster members are modeled based on the theoretical stellar model, mass function and binary properties. The field component is modeled non-parametrically using a separate field star sample in the vicinity of the cluster.


**Input**
Color-magnitude diagram of open clusters (or any other single stellar populations): allow field star contamination!


**Output**
* Parameter estimation: 
    1. logAge, [Fe/H], distance, extinction
    2. Mass function parameter: slope
    3. Binary parameters: binary fraction, mass ratio distribution
* Likelihood chain of each parameter:
    * Allow customizing priors (eg. Independent [Fe/H] measurement) to re-estimate the parameters
* Membership probability
* Bayesian Evidence: identify the open cluster candidate quantitatively

**Installation**
1. Clone the Repository
2. Install Dependencies:
    ```
    pip install h5py
    pip install dynesty
    pip install parsec_query
    pip install handy
    pip install cyper
    ```

**Usage**

1.  **Prepare Input Data in /fitting_data/**
2.  **Run the Code, see example.ipynb**

**Publication:**
MiMO: Mixture Model for Open Clusters in Color-Magnitude Diagrams
Lu Li, Zhengyi Shao, The Astrophysical Journal, Volume 930, Issue 1, id.44, 16 pp., 2022

https://ui.adsabs.harvard.edu/abs/2022ApJ...930...44L/abstract


**Contact**

For questions, issues, or collaboration opportunities, please contact:

  * **Lu Li**
  * **lilu at shao.ac.cn**
  * **Shanghai Astronomical Observatory, CAS**


