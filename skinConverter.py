from maya import cmds
import maya.api.OpenMaya as om
import pprint
import traceback
import math
import sys
import re

sys.dont_write_bytecode = True

MOVE_BONE = 2
MOVE_MATRIX = om.MMatrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, MOVE_BONE, 0, 1])
NORMALIZE = True


'''
from time import time
s=time()
run(mesh=None,  tolerance = -1)
e=time()
print('[TIME] %f sec' % (e - s))
#[TIME] 0.171000 sec
'''

def run(mesh=None,  tolerance = -1, skinHistory = False):

    if mesh:
        cmds.select(mesh)

    sel = cmds.ls(sl=1, type='transform')
    vtx_cmp, mesh = sel_component()
    if not sel and not vtx_cmp:
        cmds.warning('# Please select skin mesh.')
        return False

    procSkins = []
    if vtx_cmp:
        if skinHistory:
            procSkins = cmds.ls(cmds.listHistory(mesh), type='skinCluster')
            print('# process skinClusters : %s'%(str(procSkins)))
        weight_data = convert_weights( mesh, tolerance, vtx_cmp.getElements(), procSkins)
        set_weights( mesh, weight_data, vtx_cmp.getElements() )
    else:
        mesh = sel[0]
        if skinHistory:
            procSkins = cmds.ls(cmds.listHistory(mesh), type='skinCluster')
            print('# process skinClusters : %s'%(str(procSkins)))
        weight_data = convert_weights( mesh, tolerance, [], procSkins)
        set_weights( mesh, weight_data )
        cmds.select(mesh)


def sel_component():
    sList = om.MGlobal.getActiveSelectionList()
    dagPath, component = sList.getComponent(0)
    if component.apiTypeStr == "kMeshVertComponent":
        sIndexComp = om.MFnSingleIndexedComponent(component)
        #sIndexComp.getElements()
        return sIndexComp, dagPath.partialPathName()
    return None, None


def get_skincluster(mesh = 'pCylinder1'):
    skincl = cmds.ls(cmds.listHistory(mesh),type='skinCluster')
    if skincl:
        skincl = skincl[0]
    return skincl


def get_skin_info(skincluster):
    result = {}
    skinFn = get_dependFn(skincluster)
    mxPlug = skinFn.findPlug("matrix",0)

    ids = mxPlug.getExistingArrayAttributeIndices()
    for skinid in ids:
        idplug = mxPlug.elementByLogicalIndex( skinid )
        try:
            sObj = idplug.source()
            dagPath = om.MDagPath().getAPathTo(sObj.node())
            inf = dagPath.partialPathName()
        except: #maya2016
            attr_name = idplug.info
            inf = cmds.listConnections(attr_name, s=1, d=0)[0]
        result[inf] = skinid
    return result


def get_matrix( name='', world = True):
    args = {'q':True, 'a':True, 'm':True}
    if world:
        args['ws'] = True
    else:
        args['os'] = True
    m = cmds.xform( name,  **args)
    mm = om.MMatrix(m)
    return mm


def get_skinid(skinInfo, inf_name):
    for sinfo in skinInfo:
        if sinfo.get("inf") == inf_name:
            return sinfo.get("id")

class MoveProcess(object):

    def __init__(self, skinCluster, procSkins=[]):
        self.skinCluster = skinCluster
        self.skinInfo = get_skin_info(skinCluster)
        self.skinFn = get_dependFn(skinCluster)
        self.bpmPlug = self.skinFn.findPlug('bindPreMatrix',0)
        if not procSkins:
            procSkins = [skinCluster]
        self.procSkins = procSkins

    def set(self, inf):
        self.inf = inf
        self.skin_id   = self.skinInfo.get(inf)
        self.bpmPlugId = self.bpmPlug.elementByLogicalIndex( self.skin_id )
        self.init_mx_obj = self.bpmPlugId.asMObject()
        init_mx        = om.MFnMatrixData( self.init_mx_obj ).matrix()
        self.move_mx   = init_mx * MOVE_MATRIX

    def __enter__(self):
        mxData = om.MFnMatrixData()
        move_obj = mxData.create(self.move_mx)

        self.rebpm = {}
        if self.procSkins: # all skinCluster
            for skincl in self.procSkins:
                skinInfo = get_skin_info(skincl)
                skinFn   = get_dependFn(skincl)
                bpmPlug  = skinFn.findPlug('bindPreMatrix',0)
                skin_id  = skinInfo.get(self.inf)
                if skin_id:
                    bpmPlugId = bpmPlug.elementByLogicalIndex( skin_id )
                    init_mx_obj = bpmPlugId.asMObject()
                    bpmPlugId.setMObject(move_obj)
                    if not self.rebpm.has_key(skincl):
                        self.rebpm[skincl] = []
                    self.rebpm[skincl].append([bpmPlugId, init_mx_obj])

        for skincl in self.procSkins:
            cmds.skinCluster(skincl, edit=True, recacheBindMatrices=True)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type:
            print exception_type
        if exception_value:
            print exception_value
        if traceback:
            print traceback
        if self.rebpm:
            for skincl, procinfo in self.rebpm.items():
                for pi in procinfo:
                    pi[0].setMObject(pi[1])
        for skincl in self.procSkins:
            cmds.skinCluster(skincl, edit=True, recacheBindMatrices=True)

# Maya2016
class MoveProcess_old(object):

    def __init__(self, skinCluster, procSkins=[]):
        self.skincl = skinCluster
        self.skinInfo = get_skin_info(skinCluster)
        self.procSkins = procSkins

    def set(self, inf):
        self.inf = inf
        self.init_mx = get_matrix(name = inf, world = 1)
        self.new_mx  = self.init_mx * MOVE_MATRIX
        self.skin_id = self.skinInfo.get(inf)

    def __enter__(self):
        self.in_recncts = []
        self.out_recncts = []

        if self.procSkins: # all skinCluster
            self.joint_mx_attr = self.inf + '.worldMatrix[0]'
            skincl_attrs = cmds.listConnections(self.joint_mx_attr, s=0, d=1, p=1, type='skinCluster') or []
            for skin_at in skincl_attrs:
                skincl = skin_at.split('.')[0]
                if not skincl in self.procSkins:
                    continue

                self.skin_mx_attr  = skin_at
                cmds.disconnectAttr(self.joint_mx_attr, self.skin_mx_attr)
                cmds.setAttr(self.skin_mx_attr, *self.new_mx, type='matrix')

                self.in_recncts.append(self.joint_mx_attr)
                self.out_recncts.append(self.skin_mx_attr)

        else:  # current skinCluster
            self.skin_mx_attr  = self.skincl +'.matrix[%s]'%self.skin_id
            self.joint_mx_attr = self.inf + '.worldMatrix[0]'
            cmds.disconnectAttr(self.joint_mx_attr, self.skin_mx_attr)
            cmds.setAttr(self.skin_mx_attr, *self.new_mx, type='matrix')

            self.in_recncts.append(self.joint_mx_attr)
            self.out_recncts.append(self.skin_mx_attr)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type:
            print exception_type
        if exception_value:
            print exception_value
        if traceback:
            print traceback
        for inc, outc in zip(self.in_recncts, self.out_recncts):
            cmds.setAttr(outc, *self.init_mx, type='matrix')
            cmds.connectAttr(inc, outc)


# joint move
class MoveProcess_jnt(object):

    def __init__(self):
        pass

    def set(self, inf):
        self.inf = inf
        self.init_mx = get_matrix(name = inf, world = 1)
        self.new_mx  = self.init_mx * MOVE_MATRIX
        self.cncts = cmds.listConnections(self.inf, s=1, d=0, p=1, c=1)

    def __enter__(self):
        self.disconnect()
        cmds.xform(self.inf, ws=1, matrix=self.new_mx)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type:
            print exception_type
        if exception_value:
            print exception_value
        if traceback:
            print traceback
        cmds.xform(self.inf, ws=1, matrix=self.init_mx)
        self.connect()
        
    def disconnect(self):
        if not self.cncts:
            return
        srcAttrs, inAttrs = self.cncts[1::2], self.cncts[::2]
        for src, ina in zip(srcAttrs, inAttrs):
            cmds.disconnectAttr(src, ina)

    def connect(self):
        if not self.cncts:
            return
        srcAttrs, inAttrs = self.cncts[1::2], self.cncts[::2]
        for src, ina in zip(srcAttrs, inAttrs):
            cmds.connectAttr(src, ina, f=1)


def convert_weights( mesh, tolerance = -1, vtxids = [], procSkins = [] ):

    rest_pos    = get_vtx_pos(mesh, vtxids)
    skinCluster = get_skincluster( mesh )
    influences  = get_influences(skinCluster, vtxids)
    inf_vtx_weights = {}
    weight_data = {}

    for jt_name in influences:
        mProc = MoveProcess_jnt() # MoveProcess(skinCluster, procSkins)
        mProc.set(jt_name)
        with mProc:
            move_pos = get_vtx_pos(mesh, vtxids)
        skin_weights = calc_weights( move_pos, rest_pos )

        for vid, weight in skin_weights.items():
            if weight <= tolerance:
                continue
            if not vid in weight_data.keys():
                weight_data[vid] = []
            weight_data[vid].append( ( jt_name, weight ) )

    if NORMALIZE:
        for vid in weight_data.keys( ):
            weight_list = weight_data[ vid ]
            weight_data[vid] =  normalize_vtx_weights( weight_list )

    return weight_data


def get_meshFn(obj):
    sel = om.MSelectionList()
    sel.add( obj )
    dagPath = sel.getDagPath(0)
    meshFn = om.MFnMesh(dagPath)
    return meshFn


def get_vtx_pos(mesh, vtxids = []):
    meshFn      = get_meshFn(mesh)
    posAryDict = {}
    if not vtxids:
        for vid in xrange(meshFn.numVertices):
            posAryDict[vid] = meshFn.getPoint(vid)
    else:
        for vid in vtxids:
            posAryDict[vid] = meshFn.getPoint(vid)
    return posAryDict


def calc_vtx_dist( new_pos, rest_pos ):
    diff_pos = new_pos - rest_pos
    return math.sqrt( diff_pos.x**2 + diff_pos.y**2 + diff_pos.z**2 )


def get_dependFn(node):
    nodeSel = om.MGlobal.getSelectionListByName(node)
    mObj = nodeSel.getDependNode(0)
    dependFn = om.MFnDependencyNode(mObj)
    return dependFn


def get_influences(skincluster, vtxids = []):
    inf_list = cmds.skinCluster(skincluster,q=1,inf=1)
    if not vtxids:
        return inf_list

    skinFn = get_dependFn(skincluster)
    wlPlug = skinFn.findPlug('weightList',0)
    wPlug  = skinFn.findPlug('weights',0)
    wlAttr = wlPlug.attribute()

    result = []
    for vid in vtxids:
        wPlug.selectAncestorLogicalIndex(vid, wlAttr)
        wInfIds = wPlug.getExistingArrayAttributeIndices()
        for wInfId in wInfIds:
            result.append(inf_list[wInfId])
    return list(set(result))

##
#    skinWeight Process
#
def calc_weights( new_positions, rest_positions ):

    dict_of_weights = {}

    for vid in rest_positions.keys():
        weight = calc_vtx_dist( new_positions[vid], rest_positions[vid] ) / MOVE_BONE
        weight = round( weight, 3 )
        dict_of_weights[vid] =  weight

    return dict_of_weights

def remove_all_weighting( skincluster ):
    skinFn = get_dependFn(skincluster)
    swlPlug = skinFn.findPlug('weightList',0)

    delCL = cmds.createNode('weightGeometryFilter')
    wgfFn = get_dependFn(delCL)
    wgfPlug = wgfFn.findPlug('weightList',0)

    swlPlug.setMObject(wgfPlug.asMObject())
    cmds.delete(delCL)
    return True

def remove_vtx_weights( skincluster, vtxids=[] ):
    if not vtxids:
        remove_all_weighting( skincluster )
        return True
        # oGeoPlug = skinFn.findPlug('outputGeometry',0)
        # geoPlug  = oGeoPlug.elementByLogicalIndex( 0 )
        # meshFn   = om.MFnMesh(geoPlug.asMObject())
        # vtxids   = xrange(meshFn.numVertices)
        ## vtxids = xrange(wlPlug.numElements())

    skinFn = get_dependFn(skincluster)
    wlPlug = skinFn.findPlug('weightList',0)
    wPlug  = skinFn.findPlug('weights',0)
    wlAttr = wlPlug.attribute()
    wAttr  = wPlug.attribute()

    inf_num = xrange(len(get_influences(skincluster)))
    for vid in vtxids:
        for infId in inf_num:
            wPlug.selectAncestorLogicalIndex(int(vid), wlAttr)
            infPlug = om.MPlug(wPlug)
            infPlug.selectAncestorLogicalIndex(infId, wAttr)
            infPlug.setDouble(0)
    return True


def normalize_vtx_weights( weight_list ):

    total = sum([w for jt, w in weight_list])
    normalized_weights = []
    for bone_name, weight in weight_list:
        if weight == 0:
            normalized_weight = weight
        else:
            normalized_weight = weight / total

        normalized_weights.append( ( bone_name, normalized_weight ) )

    return normalized_weights


def set_weights( mesh, weight_data = None, vtxids=[] ):

    skincluster = get_skincluster( mesh )
    if not skincluster:
        return False

    cmds.skinPercent( skincluster, mesh, normalize=0 )
    remove_vtx_weights( skincluster , vtxids)

    skinFn = get_dependFn(skincluster)
    wlPlug = skinFn.findPlug('weightList',0)
    wPlug  = skinFn.findPlug('weights',0)
    wlAttr = wlPlug.attribute()
    wAttr  = wPlug.attribute()

    inf_list = get_influences(skincluster)
    for vert_id in weight_data.keys( ):
        weight_list = weight_data[ vert_id ]
        for inf_name, value in weight_list:
            infId = inf_list.index(inf_name)
            wPlug.selectAncestorLogicalIndex(int(vert_id), wlAttr)
            infPlug = om.MPlug(wPlug)
            infPlug.selectAncestorLogicalIndex(infId, wAttr)
            infPlug.setDouble(value)
