The code in this package implements grayscale and color image denoising as described in the paper:    
  
  Stamatis Lefkimmiatis    
  Non-Local Color Image Denoising with Convolutional Neural Networks    
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, Hawaii, July 2017.    
  
Please cite the paper if you are using this code in your research.    
Please see the file LICENSE.txt for the license governing this code.    
  
  
Overview    
————————    
The function NLNET_DENOISE_DEMO demonstrates grayscale and color image denoising with the learned models from the paper, which can all be found in the folder “network/models”.  The paper and supplementary material are provided in the folder "paper".    
  
The function NLNET_BSDS_VALIDATION_RESULTS can be used to obtain the results on the validation set extracted from the BSDS500 dataset for each one of the trained models.     
  
**Note:** The results from this function differ slightly to the reported results in the CVPR paper since they are obtained with a slightly modified network architecture which provides on average a 0.1 dB PSNR increase in restoration quality for color images compared to the original network architecture. In particular the average results obtained with the original and the newest non-local networks are provided below     
  
Noise σ (std) | CNLNET_5x5 (original) | CNLNET_5x5 (new)    
	15		33.69				**33.81**    
	25		30.96				**31.08**    
	50		27.64				**27.73**    
  
  
  
The folder “matlab/custom_layers” contains all the CNN layers that are used to build the non-local networks described in the CVPR paper, while the folder “matlab/+misc” includes some miscellaneous functions. The folder “matlab/custom_mex” includes cpu and gpu mex files used to define some of the layers of the Non-local networks.     
  
Dependencies    
————————    
The provided code has dependencies on the MatConvnet toolbox. The necessary functions are included in the folders “matlab/vl_layers”, “matlab/mex”, “matlab/src” and “matlab/compatibility”.     
  
  
Contact    
————————    
If you have questions, problems with the code, or found a bug, please let us know.    
Contact Stamatis Lefkimmiatis at s.lefkimmiatis@skoltech.ru    
