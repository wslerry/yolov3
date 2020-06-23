# -*- coding: utf-8 -*-
#-----------------------------------------------------------
# Copyright (C) 2019 Lerry William Seling
#-----------------------------------------------------------
#
# licensed under the terms of GNU GPL 3
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, see <https://www.gnu.org/licenses/>.
#
#---------------------------------------------------------------------

from osgeo import gdal
import os
import glob
# import rasterio as rio


def img2TFW(input_img, output_img):
    src = gdal.Open(input_img)
    xform = src.GetGeoTransform()
    src = None
    edit1 = xform[0] + xform[1] / 2
    edit2 = xform[3] + xform[5] / 2

    tfw = open(os.path.splitext(output_img)[0] + '.tfw', 'wt')
    tfw.write("%0.10f\n" % xform[1])
    tfw.write("%0.10f\n" % xform[2])
    tfw.write("%0.10f\n" % xform[4])
    tfw.write("%0.10f\n" % xform[5])
    tfw.write("%0.10f\n" % edit1)
    tfw.write("%0.10f\n" % edit2)
    tfw.close()

    # from affine import loadsw
    #
    # # Read from World File
    # with open(os.path.splitext(output_img)[0] + '.tfw') as tfw:
    #     loadsw(tfw.read())
    #     tfw.write("%0.10f\n" % xform[1])
    #     tfw.write("%0.10f\n" % xform[2])
    #     tfw.write("%0.10f\n" % xform[4])
    #     tfw.write("%0.10f\n" % xform[5])
    #     tfw.write("%0.10f\n" % edit1)
    #     tfw.write("%0.10f\n" % edit2)
    #     tfw.close()


    # # using rasterio
    # with rio.open(input_img) as src:
    #     affine = src.transform
    #
    # tfw_filename = input_img[:-4] + '.tfw'
    # # create tfw file
    # tfw = open(tfw_filename, 'wt')
    # tfw.write("%0.10f\n" % affine[0])
    # tfw.write("%0.10f\n" % affine[1])
    # tfw.write("%0.10f\n" % affine[3])
    # tfw.write("%0.10f\n" % affine[4])
    # tfw.write("%0.10f\n" % affine[2])
    # tfw.write("%0.10f\n" % affine[5])
    # tfw.close()


# based on https://gis.stackexchange.com/questions/9421/creating-tfw-and-prj-files-for-folder-of-geotiff-files#9426
def imgfolder2TFW(input_path):
    for infile in glob.glob(os.path.join(input_path, '*.tif')):
        src = gdal.Open(infile)
        xform = src.GetGeoTransform()
        src = None
        edit1 = xform[0] + xform[1]/2
        edit2 = xform[3] + xform[5]/2

        try:
            tfw = open(os.path.splitext(infile)[0] + '.tfw', 'wt')
            tfw.write("%0.10f\n" % xform[1])
            tfw.write("%0.10f\n" % xform[2])
            tfw.write("%0.10f\n" % xform[4])
            tfw.write("%0.10f\n" % xform[5])
            tfw.write("%0.10f\n" % edit1)
            tfw.write("%0.10f\n" % edit2)
            tfw.close()
        except Exception:
            pass
