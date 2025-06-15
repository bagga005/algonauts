import nibabel as nib
import numpy as np
import utils
# https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti
parcel_fn="/home/bagga005/algo/comp_data/brain/Schaefer2018_1000Parcels_7Networks_order.dscalar.nii"
schaefer_LR64k = nib.load(parcel_fn).get_fdata().squeeze()

def map_parcel_to_full(input, parcellation_LR64k, fill=0):
    output = np.zeros_like(parcellation_LR64k)#-100
    output[:] = fill
    for i in range(input.shape[0]): # range(1000): 0 ...999
        output[parcellation_LR64k==i+1] = input[i];
    return output

import os
inputfile = 'accuracy-s1-raw_words'
desc = 'raw words'
root_data_dir = os.path.join(utils.get_data_root_dir(), 'brain')
acc_file = os.path.join(root_data_dir, f'{inputfile}.npy')
output_file = os.path.join(root_data_dir,'visualize', f'{inputfile}_brain.png')
input = np.load(acc_file)
output = map_parcel_to_full(input, schaefer_LR64k, fill=np.nan)
print(output.shape)

# Plotting functions
from surfplot import Plot
from brainspace.utils.parcellation  import map_to_labels, reduce_by_labels
from neuromaps.datasets import fetch_fslr
import io, PIL

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
import hcp_utils as hcp

# def show_pdata(data, is_32k=False, cmap="coolwarm", mw_val=0, title=None, plot_kwargs = {}, **kwargs):
#   out = data
#   lcd = hcp.left_cortex_data(out) if not is_32k else out;
#   pkw = dict(surf_lh=lh, size=(800, 300), zoom=1.7)
#   pkw.update(plot_kwargs)
#   p = Plot(**pkw)
#   lkw = dict(cmap=cmap, cbar=True);
#   lkw.update(kwargs)
#   p.add_layer({'left': lcd},  **lkw)
#   fig = p.build()
#   if not(title is None): fig.axes[0].set_title(title)
#   return fig

# show_pdata(output, cmap="plasma", is_32k=False, plot_kwargs=dict(size=(350,130)))


print(output.shape)

import cortex
cortex.database.default_filestore = f"{root_data_dir}/pycortex_filestore"
cortex.db.filestore = f"{root_data_dir}/pycortex_filestore"
cortex.db = cortex.database.Database(f"{root_data_dir}/pycortex_filestore")
#print(cortex.database.default_filestore)

def plot_flatmap(data, mask=None, mtype = "fsaverage", height=250, data_style=None, **kwargs):
  vdata = data
  if not(mask is None):
    vdata = np.zeros_like(mask)
    vdata[mask] = data
  
  if data_style=="fullHCP":
    import hcp_utils as hcp
    vdata = np.concatenate([hcp.left_cortex_data(vdata), hcp.right_cortex_data(vdata)]);

  vertex_data = cortex.Vertex(vdata, mtype,**kwargs)
  #cortex.webshow(vertex_data, with_rois=0, height =height)
  return cortex.quickshow(vertex_data, with_rois=0, height =height);

plt = plot_flatmap(output, mtype="HCP_S1200", data_style="None", cmap="plasma")
plt.suptitle(f'{desc}', fontsize=6, y=0.99)
# plt.text(0.02, 0.02, f'{desc}', transform=plt.gca().transAxes, 
#          fontsize=6, verticalalignment='bottom')
plt.savefig(output_file)
plt.savefig('brain_visualize.png')
#plt.title('example subsurface data')
#plt.show()