##
from Diagnostic import Diagnostic
import numpy as np
from data_classes import Diag_IndVar
import MDSplus as mds

class MANITS_Camera(Diagnostic):

    def __init__(self, shotnb, cameranb):
        super().__init__('vds_mts', shotnb)
        self.contact.openTree(self.treeadrs, self.shotnb)
        self.cameranb = cameranb
        self.root_adr = '.MANTIS.MTS_CAM'+("{:02d}".format(cameranb))
        self.image_node = self.root_adr+'.FRAMES'
        self.times = Diag_IndVar(np.array(self.contact.get(self.root_adr+':TIMESTAMPS')/10.**6))
        self.exptimes = self._fetch_meta('exposure_us')/10.**6;
        self.log_gains = self._fetch_meta('gain')
        self.gains = 10**(self.log_gains/20.)
        self.nbframes = self.times.vals.size
        self.aoirow_offset = self._fetch_meta('ROI_OFFSET:Y')
        self.aoicolumn_offset = self._fetch_meta('ROI_OFFSET:X')
        self.row_size = self._fetch_meta('ROI_SIZE:Y')
        self.column_size = self._fetch_meta('ROI_SIZE:X')
        self.atomic_line = self._fetch_meta('atom_line')
        self.image_memory = np.zeros((self.row_size,self.column_size,self.nbframes),float)
        self.has_loaded = np.zeros((self.nbframes,1),bool)

        return

    def _fetch_meta(self,rel_adrs):
        fuladrs = self.root_adr+'.'+rel_adrs
        return self.contact.get(fuladrs)

    def _group_time_ids(self,id):
        if len(id) == 0:
            return []
        idxs = np.array(id)
        dels = np.diff(idxs)
        chs = dels == 1

        j = 0
        grpd_tids = []
        inds = np.argwhere(~chs)

        for adrs in inds:

            grpd_tids.append(np.array(idxs[j:(adrs[0] + 1)]))
            j = adrs[0] + 1
        grpd_tids.append(np.array(idxs[j:]))
        return grpd_tids



    def images(self, *args, **kwargs):
        specfic_times = []
        bnd_times = np.array([])
        for arg in args:
            specfic_times.append(arg)

        specfic_times = np.array(specfic_times)
        if 'btw' in kwargs:
            bnd_times = np.array(kwargs['btw'])
            if bnd_times.ndim == 1:
                bnd_times = bnd_times.reshape(1, -1)

        itdx, used_times = self.times(specfic_times,btw =bnd_times)

        previous_loaded = np.squeeze(self.has_loaded[np.squeeze(itdx)])
        vals_to_load = itdx[~previous_loaded]
        #print(vals_to_load)
        list_of_timesegs = self._group_time_ids(vals_to_load)

        self.contact.openTree(self.treeadrs,self.shotnb)
        for seg in list_of_timesegs:

            if len(seg) == 1:


                evalstr = "getSegment({},{})".format(self.image_node,seg[0])
                tmpval = np.transpose(np.array(self.contact.get(evalstr)),(1,2,0))
                self.image_memory[:,:,seg[0]:seg[0]+1] = tmpval
                self.has_loaded[seg[0]] = 1

            else:

                evaltime = "setTimeContext({},{},0)".format(seg[0]-1,seg[-1])

                self.contact.get(evaltime)
                a = np.transpose(self.contact.get(self.image_node), (1, 2, 0))
                self.image_memory[:,:,seg[:]] = np.transpose(self.contact.get(self.image_node),(1,2,0))
                self.has_loaded[seg[:]] = 1


        self.contact.closeTree(self.treeadrs, self.shotnb)
        return (self.image_memory[:,:,itdx],used_times)



