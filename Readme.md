# Substantial Evidence Suggests Strong Seismic Potential of the Qiaojia-Dongchuan Seismic Gap

This repo includes codes and results for paper **[Substantial Evidence Suggests Strong Seismic Potential of the Qiaojia-Dongchuan Seismic Gap] (#)**.
Cite the paper:
```

[Substantial Evidence Suggests Strong Seismic Potential of the Qiaojia-Dongchuan Seismic Gap] (#)
```
<!--more-->

## Abstract

The reevaluation of seismic hazards in the Xianshuihe–Xiaojiang fault zone, a previously quiet seismic zone, has become a central focus for seismology and geodynamics communities following the completion of the Baihetan Dam. This study applies a deep learning–based workflow to construct a high-resolution catalog on the dataset collected by our research group. The results reveal a previously unrecognized dipping structure at 10–16 km depth, coexisting with the dominant NNW-striking Xiaojiang Fault. Depth profiles the presence of an active asperity capable of accumulating strain energy. The spatiotemporal analysis suggests a remarkable correlation between seismic activities and the impoundment phases of the Baihetan Reservoir, providing preliminary evidence of reservoir-triggered stress perturbations. High Coulomb Failure Stress rates and earthquake triggering probabilities with decreasing b-value in the Qiaojia–Dongchuan segment highlight its role as a stress-loading seismic gap with elevated rupture potential under current regional stress conditions. The study contributes new insights into fault mechanics, reservoir effects, and seismic hazard estimation.

## Requirments

fastapi <br>obspy<br>uvicorn<br>pandas

### Repo structure

```plaintext

Qiaojia/
├── gamma2HypoDD/
│   ├── HYPODD/
│   ├── tmp_00/
│   ├── hypo_catalog_*.txt          # the final catalog resulted by HypoDD
│   ├── convert_stations.py
│   └── run_gamma2hypodd.py 
├── gamma2hypoinverse/
│   ├── HYPODD/
│   ├── hyp1.40/
│   ├── convert_stations.py
│   ├── gamma2hypoinverse.py 
│   ├── hypoinverse2hypodd.py     
│   ├── run.sh                # final prompt
│   ├── hyp.command         # the parameters fro hypoinverse
│   └── hypoDD_v*_*_withmag.reloc    # the final catalog resulted by Hypoinverse followed by HypoDD
├── GammaAssociation/       
│   ├── run_gamma_original.ipynb       # the prompt of Gamma
│   └── station_lists.txt             # the location of stations
├── postprocessing/  
│   ├── recalculated magnitude.ipynb     # calculate the magnitude using Yang's
│   └── MT图.ipynb
├── preprocessing/             
│   ├── file_reader.ipynb     
│   ├── test.ipynb           
│   └── merge_picks.ipynb     
├── ReadMe.md
├── test.ipynb    
├── seisbench.ipynb              # Phasenet and EQTransformer detection
└── inference_final.py           # Transeis detection
```

### Acknowledgement

Find the [GNSS velocity and strain rate data](https://zenodo.org/records/10215151) used in this study. We are grateful to the ZMAP team for their development of the [ZMAP](https://github.com/CelsoReyes/zmap7) tool.
