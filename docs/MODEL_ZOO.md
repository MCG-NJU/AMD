# AMD Model Zoo
### Something-Something V2

|  Method  | Extra Data | Backbone | Epoch | \#Frame |                          Pre-train                           |                          Fine-tune                           | Top-1 | Top-5 |
| :------: | :--------: | :------: | :---: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :---: |
| AMD |  ***no***  |  ViT-S   | 800  | 16x2x3  | [log](https://drive.google.com/file/d/1gNCaxIuzRfVuETS_UFLwCSLqr0zej0Tt/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/12hEVVdrFv0VNAIqAIZ0UTeicn-tm4jva/view?usp=drive_link) | [log](https://drive.google.com/file/d/1Y0hDs8uMDr5Hs0o3546uV5H2hA7TyF9L/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/1ynDyu3K_INoZjaNLzFIaYCo6kjBOwG__/view?usp=sharing) | 70.2 | 92.5 |
| AMD |  ***no***  |  ViT-B   | 800  | 16x2x3  | [log](https://drive.google.com/file/d/13rYTQZ-AQYSWTpA0yD0jQm6d_k64MDxx/view?usp=sharing)/[checkpoint](https://drive.google.com/file/d/13EOb--vymBQpLNbztcN7wkdxXJiVfyXa/view?usp=sharing) | [log](https://drive.google.com/file/d/1sX5nu92fg5LxbZ8_oqJoAc2Y1g_tDqc3/view?usp=drive_link)/[checkpoint](https://drive.google.com/file/d/1zc3ITIp4rR-dSelH0KaqMA9ahyju7sV2/view?usp=drive_link) | 73.3 | 94.0 |
### Kinetics-400

|  Method  | Extra Data | Backbone | Epoch | \#Frame |                          Pre-train                           |                          Fine-tune                           | Top-1 | Top-5 |
| :------: | :--------: | :------: | :---: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :---: |
| AMD |  ***no***  |  ViT-S   | 800  | 16x5x3  | [log](https://drive.google.com/file/d/198tthnBARZwU39gKWSjhBtTJmA8ipPJB/view?usp=drive_link)/[checkpoint](https://drive.google.com/file/d/1UVmFl_sdoSPYvhvMfdaFhghUOGZVFUCi/view?usp=sharing) | [log](https://drive.google.com/file/d/1aiqlMO1SsAn7sTmvtFXKlpmY_rrw-BaJ/view?usp=drive_link)/[checkpoint](https://drive.google.com/file/d/1dJZKPHD9bYUl0UKH2qLca5JSTengJ8nq/view?usp=drive_link) | 80.1 | 94.5 |
| AMD |  ***no***  |  ViT-B   | 800  | 16x5x3  | [log](https://drive.google.com/file/d/1JO9dxDwU0pByh9Xu0RXc9s8w1-iEiMsD/view?usp=drive_link)/[checkpoint](https://drive.google.com/file/d/1zTJxxqHRN7ZiSfNiqDophTKvv2-Kh08F/view?usp=drive_link) | [log](https://drive.google.com/file/d/1GJCyRWAKKx6Fepka6UuQHl-rxBQ2Wbm1/view?usp=drive_link)/[checkpoint](https://drive.google.com/file/d/1zVMMVQoODUa_OlC40d8lvI1fgZPoqPjd/view?usp=drive_link) | 82.2 | 95.3 |
### ImageNet-1K
|  Method  | Extra Data | Backbone | Epoch  |                          Pre-train                           |                          Fine-tune                           | Top-1 |
| :------: | :--------: | :------:  | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: |
| AMD |  ***no***  |  ViT-S   | 800  | [log](https://drive.google.com/file/d/1_6ZeX5flzqxHbYbEA9bnfuKpdRjZmVBZ/view?usp=drive_link)/[checkpoint](https://drive.google.com/file/d/1_YOnTT14zllIMZ-kte6Uwj0Kwp2FW-AX/view?usp=drive_link) | [log](https://drive.google.com/file/d/1D48g24mNJjwUXG3dxqa5u989bxYIggJ9/view?usp=drive_link)/[checkpoint](https://drive.google.com/file/d/1CjMPAht9EqrewMpagRTyOcOmaG3I_6t4/view?usp=sharing) | 82.1 |
| AMD |  ***no***  |  ViT-B   | 800  | [log](https://drive.google.com/file/d/1f8hfDCj2nVzjgKOreypYD-8uFewizbq1/view?usp=drive_link)/[checkpoint](https://drive.google.com/file/d/1a5DduUSPo9Z3oPYej7Q3qojvNt5cBX7z/view?usp=drive_link) | [log](https://drive.google.com/file/d/1x7t_cXXABw_2dmC1CV_x8WQEi9htse7G/view?usp=drive_link)/[checkpoint](https://drive.google.com/file/d/1SiORgNWFR5siETjAKgcgL03yGUGb1j8y/view?usp=drive_link) | 84.6 |
### Note:

- We report the results of AMD finetuned with `I3D dense sampling` on **Kinetics-400** and `TSN uniform sampling` on **Something-Something V2**, respectively.
- \#Frame = #input_frame x #clip x #crop.
- \#input_frame means how many frames are input for model during the test phase.
- \#crop means spatial crops (e.g., 3 for left/right/center crop).
- \#clip means temporal clips (e.g., 5 means repeted temporal sampling five clips with different start indices).