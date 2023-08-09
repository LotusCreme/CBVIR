# CBVIR
Full name: Content-based Video Image Retrieval System (CBVIR-S)

Introduction:
CBVIR-S is a system that retrieves a given query image from a given video, also providing the temporal indices of corresponding retrieval results. 

How to use it:









Author:
Master students in TU Delft EEMCS. 

Institution:
The testing dataset of CBVIR-S is provided by Andrea Natetti in Nanyang Technological University, School of Art, Design and Media. 

License:
CBVIR-S and all the codes are released under MIT License. The MIT License grants users the right to use, copy, modify, merge, distribute and sublicense, free of charge.


# KFE module


# GUI prototype
To use the GUI prototype, you can open the file "demo.py" in "CBVIR_GUI". By simply uploading your query image and video, you will get the temporal location of the target image. 

![image](https://github.com/LotusCreme/CBVIR/assets/141781811/27728bcd-ae09-43e5-a52d-793c23af45ff | width=100)
![He1results1](https://github.com/LotusCreme/CBVIR/assets/141781811/fb53b55b-804b-4ee0-a38c-ef2103aad6f7)
![He11result1](https://github.com/LotusCreme/CBVIR/assets/141781811/666146e4-8c81-4c0e-a18e-453452257536)
![Polo1results2](https://github.com/LotusCreme/CBVIR/assets/141781811/f1aeff71-87d8-4907-9505-6a79e38c4dd0)
![Ba4result1](https://github.com/LotusCreme/CBVIR/assets/141781811/305e227d-e2bb-483c-bf52-1f2614cfe7e5)


If the performance of certain image and video pair is not satisfactory, you could also adjust the "similarity threshold" in utils/KFE_module or "thr" in utils/CBIR_module. These two parameters will be adjustable in "Settings" in the GUI prototype in the near future. :)
