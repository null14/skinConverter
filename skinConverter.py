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
run(mesh=None,  tolerance = 0)
e=time()
print('[TIME] %f sec' % (e - s))
#[TIME] 0.171000 sec
'''

def run(mesh=None,  tolerance = -1):

    if mesh:
        cmds.select(mesh)

    sel = cmds.ls(sl=1, type='transform')
    vtx_cmp, mesh = sel_component()
    if not sel and not vtx_cmp:
        cmds.warning('# Please select skin mesh.')
        return False

    if vtx_cmp:
        weight_data = convert_weights( mesh, tolerance, vtx_cmp.getElements())
        set_weights( mesh, weight_data, vtx_cmp.getElements() )
    else:
        mesh = sel[0]
        weight_data = convert_weights( mesh, tolerance, [])
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
        sObj = idplug.source()
        dagPath = om.MDagPath().getAPathTo(sObj.node())
        inf = dagPath.partialPathName()
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

    def __init__(self, skinCluster):
        self.skinInfo = get_skin_info(skinCluster)
        skinFn        = get_dependFn(skinCluster)
        self.bpmPlug  = skinFn.findPlug('bindPreMatrix',0)

    def set(self, inf):
        self.skin_id   = self.skinInfo.get(inf)
        self.bpmPlugId = self.bpmPlug.elementByLogicalIndex( self.skin_id )
        self.init_mx_obj = self.bpmPlugId.asMObject()
        init_mx        = om.MFnMatrixData( self.init_mx_obj ).matrix()
        self.move_mx   = init_mx * MOVE_MATRIX

    def __enter__(self):
        mxData = om.MFnMatrixData()
        move_obj = mxData.create(self.move_mx)
        self.bpmPlugId.setMObject(move_obj)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type:
            print exception_type
        if exception_value:
            print exception_value
        if traceback:
            print traceback
        self.bpmPlugId.setMObject(self.init_mx_obj)


class MoveProcess_old(object):

    def __init__(self, skinCluster):
        self.skincl = skinCluster
        self.skinInfo = get_skin_info(skinCluster)

    def set(self, inf):
        self.inf = inf
        init_mx  = get_matrix(name = inf, world = 1)
        self.new_mx  = init_mx * MOVE_MATRIX
        for sinfo in self.skinInfo:
            if sinfo.get("inf") == self.inf:
                self.skin_id = sinfo.get("id")

    def __enter__(self):
        self.skin_mx_attr  = self.skincl +'.matrix[%s]'%self.skin_id
        self.joint_mx_attr = self.inf + '.worldMatrix[0]'
        cmds.disconnectAttr(self.joint_mx_attr, self.skin_mx_attr)
        cmds.setAttr(self.skin_mx_attr, *self.new_mx, type='matrix')
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type:
            print exception_type
        if exception_value:
            print exception_value
        if traceback:
            print traceback
        cmds.connectAttr(self.joint_mx_attr, self.skin_mx_attr)

def convert_weights( mesh, tolerance = -1, vtxids = [] ):

    rest_pos    = get_vtx_pos(mesh, vtxids)
    skinCluster = get_skincluster( mesh )
    influences  = get_influences(skinCluster, vtxids)
    inf_vtx_weights = {}
    weight_data = {}

    for jt_name in influences:
        mProc = MoveProcess(skinCluster)
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
