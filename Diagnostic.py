##
from MDSplus.connection import Connection
import MDSplus as mds



class Diagnostic():

    def __init__(self,tree_adrs,shotnb,connect_adrs = 'localhost:5555'):
        self.contact = Connection(connect_adrs)
        self.treeadrs =tree_adrs
        self.shotnb = shotnb


    def time(self):
        pass

