#*******************************************************************************

import os, json, shapefile

import numpy                  as np
import matplotlib             as mpl
import matplotlib.pyplot      as plt

import matplotlib.patches     as patch
import matplotlib.lines       as line

import matplotlib.cm          as cm

from sklearn.cluster import KMeans

#*******************************************************************************

def vizShape(shpdat=None, axHndl=None, clr=[0.0,0.5,0.0]):

  skipnext = 0
  shpbrk   = list(shpdat.parts).copy()
  shpbrk.append(len(shpdat.points))

  for k1 in range(1,len(shpbrk)):
    if(skipnext > 0):
      skipnext = skipnext - 1
      continue

    subdat = np.array(shpdat.points[shpbrk[k1-1]:shpbrk[k1]])

    cpos   = k1

    while(cpos < len(shpbrk)-1):

      subdatadd = np.array(shpdat.points[shpbrk[cpos]:shpbrk[cpos+1]])

      pn        = np.sum((subdatadd[1:,0]-subdatadd[:-1,0])*
                         (subdatadd[1:,1]+subdatadd[:-1,1]))
      if(pn<0):
        subdat = np.vstack((subdat,subdatadd,subdat[-1,:]))
        skipnext = skipnext + 1;
        cpos = cpos + 1;
      else:
        break

    polyShp = patch.Polygon(subdat, fc=clr, ec=clr, lw=1.0, ls='-')
    axHndl.add_patch(polyShp)

  return None

#*******************************************************************************

def vizOutline(shpdat=None, axHndl=None, clr=[0.0,0.0,0.0], wid=0.5):

  shpbrk   = list(shpdat.parts).copy()
  shpbrk.append(len(shpdat.points))

  for k1 in range(1,len(shpbrk)):
    subdat  = np.array(shpdat.points[shpbrk[k1-1]:shpbrk[k1]])
    lineShp = line.Line2D(subdat[:,0],subdat[:,1],linewidth=wid,color=clr);
    axHndl.add_line(lineShp)

  return None

#*******************************************************************************

# Calculated haversine distance
def h_dist(lat1, long1, lat2, long2):

  lat1  = np.pi*lat1 /180.0
  lat2  = np.pi*lat2 /180.0
  long1 = np.pi*long1/180.0
  long2 = np.pi*long2/180.0

  delt = 6371*2*np.arcsin(np.sqrt(np.power(np.sin(0.5*lat2-0.5*lat1),2.0) +
                     np.cos(lat2)*np.cos(lat1)*np.power(np.sin(0.5*long2-0.5*long1),2.0)))

  return delt

#*******************************************************************************


N_CLUST     = 5
DATA_ROOT   = os.path.join(os.environ['DATA_ROOT'])


with open('dump_pop.json') as fid01:
  pop_dict = json.load(fid01)

with open('dump_area.json') as fid01:
  area_dict = json.load(fid01)

with open('dump_latlong.json') as fid01:
  latlong_dict = json.load(fid01)

with open('dump_mcv.json') as fid01:
  mcv_dict = json.load(fid01)

with open('dump_cgf_stunting.json') as fid01:
  cgf_dict = json.load(fid01)




name_vec    = sorted(list(pop_dict.keys()))
distmat     = np.zeros((len(name_vec),len(name_vec)))
pop_vec     = np.array([pop_dict[val]     for val in name_vec])
area_vec    = np.array([area_dict[val]    for val in name_vec])
latlong_vec = np.array([latlong_dict[val] for val in name_vec])


for k1 in range(latlong_vec.shape[0]):
  distmat[k1,:]  = h_dist(latlong_vec[k1,0],latlong_vec[k1,1],latlong_vec[:,0],latlong_vec[:,1])



pop_vec[pop_vec==0] = 1

dvec001 = np.log(pop_vec/area_vec)
dvec001 = dvec001 - np.min(dvec001)
dvec001 = dvec001 / np.max(dvec001)
dvec001 = dvec001.reshape(-1,1)



dvec020 = np.zeros((len(name_vec),1))
for k1 in range(len(name_vec)):
  bidx = distmat[k1,:] < 20
  dvec020[k1,0] = np.sum(pop_vec[bidx])/np.sum(area_vec[bidx])

dvec020 = np.log(dvec020)
dvec020 = dvec020 - np.min(dvec020)
dvec020 = dvec020 / np.max(dvec020)
dvec020 = dvec020.reshape(-1,1)


dvec100 = np.zeros((len(name_vec),1))
for k1 in range(len(name_vec)):
  bidx = distmat[k1,:] < 100
  dvec100[k1,0] = np.sum(pop_vec[bidx])/np.sum(area_vec[bidx])

dvec100 = np.log(dvec100)
dvec100 = dvec100 - np.min(dvec100)
dvec100 = dvec100 / np.max(dvec100)
dvec100 = dvec100.reshape(-1,1)




dvec200 = np.zeros((len(name_vec),1))
for k1 in range(len(name_vec)):
  bidx = distmat[k1,:] < 200
  dvec200[k1,0] = np.sum(pop_vec[bidx])/np.sum(area_vec[bidx])

dvec200 = np.log(dvec200)
dvec200 = dvec200 - np.min(dvec200)
dvec200 = dvec200 / np.max(dvec200)
dvec200 = dvec200.reshape(-1,1)



fmat = np.hstack((dvec001, dvec020, dvec100, dvec200))
print(fmat.shape)

# Feed points interior to shape into k-means clustering to get num_box equal(-ish) clusters;
sub_clust = KMeans(n_clusters=N_CLUST, random_state=4, n_init='auto').fit(fmat)

sub_node  = sub_clust.cluster_centers_
sub_labs  = sub_clust.labels_


sub_clr   = sub_node[:,0]
sub_clr   = sub_clr - np.min(sub_clr)
sub_clr   = sub_clr / np.max(sub_clr)




# Input shapefiles
sf0    = shapefile.Reader(os.path.join(DATA_ROOT,'Shapefiles','nga_lev00_country'))
sf0s   = sf0.shapes()
sf0r   = sf0.records()

sf1    = shapefile.Reader(os.path.join(DATA_ROOT,'Shapefiles','nga_lev01_states'))
sf1s   = sf1.shapes()
sf1r   = sf1.records()

sf3    = shapefile.Reader('nga_lev02_lgas_sub_100km')
sf3s   = sf3.shapes()
sf3r   = sf3.records()



# Figure 0
fig01 = plt.figure(figsize=(10+3,10*2.5/4))
axs01 = fig01.add_subplot(111,label=None)
plt.sca(axs01)

axs01.grid(visible=True, which='major', ls='-', lw=0.5, label='')
axs01.grid(visible=True, which='minor', ls=':', lw=0.1)
axs01.set_axisbelow(True)
axs01.set_xlabel('Longitude',fontsize=14)
axs01.set_ylabel('Latitude', fontsize=14)

axs01.set_xlim(  2.5, 15.0)
axs01.set_ylim(  4.0, 14.0)

virmap   = mpl.colormaps['viridis']

for k1 in range(len(sf3r)):
  sidx = name_vec.index(sf3r[k1][0])
  cval = virmap(sub_clr[sub_labs[sidx]])[:3]
  vizShape(sf3s[k1],axs01,clr=cval)


for k1 in range(len(sf1r)):
  if('KADUNA' not in sf1r[k1][0] and 'KANO' not in sf1r[k1][0]):
    continue
  vizOutline(sf1s[k1],axs01,wid=1.0,clr=[0.6,0.2,0.0])


vizOutline(sf0s[0],axs01,wid=1.0,clr=[0.6,0.2,0.0])


cbar_handle = plt.colorbar(cm.ScalarMappable(cmap=virmap), shrink=0.75, ax=axs01)


ticloc = (np.arange(N_CLUST)/(N_CLUST-1)).tolist()
ticlab = ['{:d}'.format(k1) for k1 in range(N_CLUST)]
cbar_handle.set_ticks(ticks=ticloc)
cbar_handle.set_ticklabels(ticlab)
cbar_handle.set_label('Population Class',fontsize=20,labelpad=10)
cbar_handle.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig('fig_NGA_popclass04.png')
plt.close()



# Figure 1
fig01 = plt.figure(figsize=(10+3,10*2.5/4))
axs01 = fig01.add_subplot(111,label=None)
plt.sca(axs01)

axs01.grid(visible=True, which='major', ls='-', lw=0.5, label='')
axs01.grid(visible=True, which='minor', ls=':', lw=0.1)
axs01.set_axisbelow(True)
axs01.set_xlabel('Longitude',fontsize=14)
axs01.set_ylabel('Latitude', fontsize=14)


axs01.set_xlim(  2.5, 15.0)
axs01.set_ylim(  4.0, 14.0)

virmap   = mpl.colormaps['viridis']
pdenlist = list()

for k1 in range(len(sf3r)):

  aval = area_dict[sf3r[k1][0]]
  pval = pop_dict[sf3r[k1][0]]
  pden = pval/aval + 1e-5

  pdenlist.append(np.log(pden))
  pden = min(np.log(pden)/np.log(5000),1.0)
  cval = virmap(pden)[:3]

  if(pden<-1.3):
    cval=[0.5,0.5,0.5]

  vizShape(sf3s[k1],axs01,clr=cval)


for k1 in range(len(sf1r)):
  if('KADUNA' not in sf1r[k1][0] and 'KANO' not in sf1r[k1][0]):
    continue
  vizOutline(sf1s[k1],axs01,wid=1.0,clr=[0.6,0.2,0.0])


vizOutline(sf0s[0],axs01,wid=1.0,clr=[0.6,0.2,0.0])


cbar_handle = plt.colorbar(cm.ScalarMappable(cmap=virmap), shrink=0.75, ax=axs01)


ticloc = [0.1889,0.4593,0.72965,1.0]
ticlab = ['5','50','500','5000']
cbar_handle.set_ticks(ticks=ticloc)
cbar_handle.set_ticklabels(ticlab)
cbar_handle.set_label('Population Density ($\mathrm{km}^{-2}$)',fontsize=20,labelpad=10)
cbar_handle.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig('fig_NGA_density.png')
plt.close()


print(np.max(pdenlist))



# Figure 2
fig01 = plt.figure(figsize=(10+3,10*2.5/4))
axs01 = fig01.add_subplot(111,label=None)
plt.sca(axs01)

axs01.grid(visible=True, which='major', ls='-', lw=0.5, label='')
axs01.grid(visible=True, which='minor', ls=':', lw=0.1)
axs01.set_axisbelow(True)
axs01.set_xlabel('Longitude',fontsize=14)
axs01.set_ylabel('Latitude', fontsize=14)


axs01.set_xlim(  2.5, 15.0)
axs01.set_ylim(  4.0, 14.0)

virmap   = mpl.colormaps['viridis']
pdenlist = list()

for k1 in range(len(sf3r)):

  pden = mcv_dict[sf3r[k1][0]]['val']
  cval = virmap(pden)[:3]
  vizShape(sf3s[k1],axs01,clr=cval)


for k1 in range(len(sf1r)):
  if('KADUNA' not in sf1r[k1][0] and 'KANO' not in sf1r[k1][0]):
    continue
  vizOutline(sf1s[k1],axs01,wid=1.0,clr=[0.6,0.2,0.0])


vizOutline(sf0s[0],axs01,wid=1.0,clr=[0.6,0.2,0.0])


cbar_handle = plt.colorbar(cm.ScalarMappable(cmap=virmap), shrink=0.75, ax=axs01)

cbar_handle.set_label('MCV Coverage',fontsize=20,labelpad=10)
cbar_handle.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig('fig_NGA_MCV.png')
plt.close()



# Figure 3
fig01 = plt.figure(figsize=(10+3,10*2.5/4))
axs01 = fig01.add_subplot(111,label=None)
plt.sca(axs01)

axs01.grid(visible=True, which='major', ls='-', lw=0.5, label='')
axs01.grid(visible=True, which='minor', ls=':', lw=0.1)
axs01.set_axisbelow(True)
axs01.set_xlabel('Longitude',fontsize=14)
axs01.set_ylabel('Latitude', fontsize=14)


axs01.set_xlim(  2.5, 15.0)
axs01.set_ylim(  4.0, 14.0)

virmap   = mpl.colormaps['viridis']
pdenlist = list()

for k1 in range(len(sf3r)):

  pden = cgf_dict[sf3r[k1][0]]['val']/100.0
  cval = virmap(pden)[:3]
  vizShape(sf3s[k1],axs01,clr=cval)


for k1 in range(len(sf1r)):
  if('KADUNA' not in sf1r[k1][0] and 'KANO' not in sf1r[k1][0]):
    continue
  vizOutline(sf1s[k1],axs01,wid=1.0,clr=[0.6,0.2,0.0])


vizOutline(sf0s[0],axs01,wid=1.0,clr=[0.6,0.2,0.0])


cbar_handle = plt.colorbar(cm.ScalarMappable(cmap=virmap), shrink=0.75, ax=axs01)

cbar_handle.set_label('Stunting',fontsize=20,labelpad=10)
cbar_handle.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig('fig_NGA_stunting.png')
plt.close()
