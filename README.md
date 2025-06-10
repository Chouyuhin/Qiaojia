## requirments
fastapi <br>obspy<br>uvicorn<br>pandas



```plaintext

Qiaojia/
├── AssResults_modifydepth/     #原始picks修改灵敏度，除以灵敏度以后的振幅，深度
│   ├── assignmets_gamma_original.txt         #gamma得到的assignments
│   ├── catalog_gamma_original.txt           #gamma得到的没转换坐标的catalog
│   ├── gamma_catalog.txt      #gamma得到的最终catalog
│   └── gamma_picks.txt        # gamma得到的picks，包含event_index
├── DeepDenoiser/
│   ├── deepdenoiser/
│   │   └── predict.py
├── gamma2HypoDD/
│   ├── HYPODD/
│   ├── tmp_00/
│   ├── hypo_catalog_*.txt          # hypoDD得到的最终catalog
│   ├── convert_stations.py
│   └── run_gamma2hypodd.py 
├── gamma2hypoinverse/
│   ├── HYPODD/
│   ├── hyp1.40/
│   ├── convert_stations.py
│   ├── gamma2hypoinverse.py 
│   ├── hypoinverse2hypodd.py     
│   ├── run.sh                # 最终的执行文件
│   ├── hyp.command         # hypoinverse的参数文件
│   └── hypoDD_v*_*_withmag.reloc    # hypoinverse和hypoDD得到的最终catalog
├── GammaAssociation/       # 调用gamma以及输入文件
│   ├── merged_output.csv    # 只包含QJ02，QJ03的结果，demo  
│   ├── merged_picks.csv     # 所有台站pick合在一起的结果
│   ├── merged_pick_scaled.csv #所有台站pick合在一起并且振幅除以灵敏度调整以后的
│   ├── run_gamma_original.ipynb       # 运行gamma的执行文件
│   └── 宽频带布点.txt 
├── postprocessing/           # 后处理，画图
│   └── MT图.ipynb
├── preprocessing/             # 预处理
│   ├── cutAndcsv.ipynb       # 裁剪miniseed波形为30s，为deepdenoiser提供输入
│   ├── file_reader.ipynb     # 读文件的utils
│   ├── npztoh5.ipynb         # 将降噪得到的npztoh5拼成200s的波形并以h5保存
│   ├── test.ipynb            # 一些utils
│   └── merge_picks.ipynb     # 合并所有拾取的picks
├── ReadMe.md
├── test.ipynb                # 另一些utils
└── inference4QJ.py           # 调用Transeis为巧家
```
