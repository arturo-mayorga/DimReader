import sys
import csv
import numpy as np
import pyqtgraph as pg
from PyQt4 import QtCore,QtGui

class Grid():
    def __init__(self,points,resultVect,gridCoords):
        self.points = points
        #currently assumes the points given in a 2D array with the [n][0] being the smallest x and y and
        # [n][m] the largest (x values increase from left to right, y values from bottom to top)
        self.gridCoords = gridCoords
        self.resultVect= resultVect

    def addAvgNeighbors(self,c):
        print("avg neighb init")
        for i in range(self.nrow* self.ncol):
            c.append([])
            self.resultVect.append(0)
            for j in range(self.nrow * self.ncol):
                c[2*len(self.points)+i].append(0)
        # append necessary zeros to the results vector (one for each point on the grid)
        # for i in range(self.nrow * self.ncol):
        #     self.resultVect.append(0)
        # calculate the weights for the average of the neighbors for each vi
        for i in range(len(c[0])):
            cInd = 2 * len(self.points) + i
            c[cInd][i] = 1
            if i % self.ncol == 0:
                # it is on the left edge (no neighbors to the left)
                if i < self.ncol:
                    # it is the top left corner
                    c[cInd][i + 1] = -.5
                    c[cInd][i +self.ncol] = -.5

                elif i >= self.ncol * (self.nrow - 1):
                    # it is the bottom left corner
                    c[cInd][i + 1] = -.5
                    c[cInd][i - self.ncol] = -.5

                else:
                    # interior point on the left edge
                    c[cInd][i - self.ncol] = -1.0 / 3.0
                    c[cInd][i + 1] = -1.0 / 3.0
                    c[cInd][i + self.ncol] = -1.0 / 3.0

            elif (i + 1) % self.ncol == 0:
                # point is on the right edge (no neighbors to the right)
                if i < self.ncol:
                    # it is the top right corner
                    c[cInd][i - 1] = -.5
                    c[cInd][i + self.ncol] = -.5
                elif i >= self.ncol * (self.nrow - 1):
                    # it is the bottom right corner
                    c[cInd][i - 1] = -.5
                    c[cInd][i - self.ncol] = -.5

                else:
                    # interior point on the right edge
                    c[cInd][i - self.ncol] = -1.0 / 3.0
                    c[cInd][i - 1] = -1.0 / 3.0
                    c[cInd][i + self.ncol] = -1.0 / 3.0

            else:
                if i < self.ncol:
                    # point is an interior point on the top row
                    c[cInd][i + self.ncol] = -1.0 / 3.0
                    c[cInd][i - 1] = -1.0 / 3.0
                    c[cInd][i + 1] = -1.0 / 3.0

                elif i >= self.ncol * (self.nrow - 1):
                    c[cInd][i - self.ncol] = -1.0 / 3.0
                    c[cInd][i - 1] = -1.0 / 3.0
                    c[cInd][i + 1] = -1.0 / 3.0
                else:
                    c[cInd][i + self.ncol] = -.25
                    c[cInd][i - self.ncol] = -.25
                    c[cInd][i - 1] = -.25
                    c[cInd][i + 1] = -.25

    def addAvgNeighborsInner(self,c):
        if(self.nrow>2 and self.ncol>2):

            # calculate the weights for the average of the neighbors for each vi
            for i in range(((self.nrow) * (self.ncol))):

                if(i % self.ncol != 0 and (i + 1) % self.ncol != 0 and i<self.ncol * (self.nrow - 1) and i>self.ncol):
                    c.append([])
                    self.resultVect.append(0)
                    cInd = len(c)-1
                    for j in range(self.nrow * self.ncol):
                        c[cInd].append(0)

                    c[cInd][i] = 1
                    #print(str(int(i / self.nrow)) + ", " + str(i % self.nrow))
                    c[cInd][i + self.ncol] = -.25
                    c[cInd][i - self.ncol] = -.25
                    c[cInd][i - 1] = -.25
                    c[cInd][i + 1] = -.25


    def calcGridPoints(self):

        #assumes grid is rectangular (equal number of squares in each row)
        self.nrow = len(self.gridCoords)
        self.ncol = len(self.gridCoords[0])
        c = []
        #initialize c (first n rows are for x derivatives, the second n are for y derivative, the rest
        #are for the average neighbors of each corner)
        print("inits")
        for i in range(2*len(self.points)):
            c.append([])
            for j in range(self.nrow*self.ncol):
                c[i].append(0)


        corners = []

        #iterate over each square
        for i in range(self.nrow-1):
            corners.append([])
            for j in range(self.ncol-1):
                corners[i].append([self.gridCoords[i][j+1],self.gridCoords[i][j],self.gridCoords[i+1][j],self.gridCoords[i+1][j+1]])

        print("corners")
        for k in range(len(self.points)):
            # print(k)
            p = self.points[k]
            i = 0
            j =0
            found = False
            #max iterations is max(num rows, num cols)
            while(not found):
                #find the maxX,minX, miny and maxY of the corners
                maxX = corners[i][j][0][0]
                minX = corners[i][j][0][0]
                maxY= corners[i][j][0][1]
                minY= corners[i][j][0][1]
                for l in range(1,3):
                    if(corners[i][j][l][0]>maxX):
                        maxX= corners[i][j][l][0]
                    elif (corners[i][j][l][0] < minX):
                        minX = corners[i][j][l][0]
                    if (corners[i][j][l][1] > maxY):
                        maxY = corners[i][j][l][1]
                    if (corners[i][j][l][1] < minY):
                        minY = corners[i][j][l][1]

                #check if it is in the current square
                if (p[0]<=maxX or abs(p[0]-maxX)<=pow(10,-5) )and (p[0]>=minX or abs(p[0]-minX)<=pow(10,-5)) and (p[1]>=minY or abs(p[1]-minY)<=pow(10,-5))and (p[1]<=maxY or abs(p[1]-maxY)<=pow(10,-5)):

                    xscaled = (p[0]-corners[i][j][2][0])/(corners[i][j][3][0]-corners[i][j][2][0])
                    yscaled = (p[1]-corners[i][j][2][1])/(corners[i][j][1][1]-corners[i][j][2][1])


                    if (xscaled >= yscaled):
                        # point is in lower triangle
                        weightYv3=0
                        weightYv2 = -1.0
                        weightYv1 = 1.0

                        weightXv3 = -1.0
                        weightXv2 = 1.0
                        weightXv1 = 0

                        c[2*k][(i+1)*self.ncol+j+1] = weightXv2
                        c[2*k+1][(i+1)*self.ncol+j+1] = weightYv2

                    else:
                        weightYv1 = 0
                        weightYv3 = -1.0
                        weightYv2 = 1.0

                        weightXv3 = 0
                        weightXv2 = -1.0
                        weightXv1 = 1.0

                        c[2*k][i * self.ncol + j] = weightXv2
                        c[2*k+1][i*self.ncol+j] = weightYv2
                    c[2*k][i * self.ncol + (j + 1)] = weightXv1
                    c[2*k][(i + 1) * self.ncol + j] = weightXv3
                    c[2*k+1][i * self.ncol + (j + 1)] = weightYv1
                    c[2*k+1][(i + 1) * self.ncol + j] = weightYv3
                    found = True
                    #if it is of smaller y value, add to row counter
                elif p[1]<minY:# or p[1]>maxY:
                    # if it is of greater x value, add to col counter
                    if(p[0]>maxX):# or p[0]<minX):
                        j+=1
                    i += 1
                # if it is of greater x value, add to col counter
                elif (p[0]>maxX):# or p[0]<minX):
                    j += 1

        self.addAvgNeighbors(c)

        #Perform matrix calculations to get V0,...,Vn
        c = np.array(c)
        resultVect = np.array(self.resultVect)

        self.vVect = np.linalg.lstsq(c,resultVect,rcond=0.000001)

        return self.vVect[0]

class Edge:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def getEdge(self):
        return [(self.x1, self.y1), (self.x2, self.y2)]

    def updateEdge(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def getP1(self):
        return [self.x1, self.y1]

    def getP2(self):
        return [self.x2, self.y2]

    def getPointList(self):
        return [[self.x1, self.y1], [self.x2, self.y2]]

    def __str__(self):
        return "[(" + str(self.x1) + ", " + str(self.y1) + "),(" + str(self.x2) + "," + str(self.y2) + ")]"

    def copy(self):
        return Edge(self.x1, self.y1, self.x2, self.y2)

class MarchingSqaures:



    def __init__(self,gridPoints,isoval,gridCoord=None):

        #a 2D array storing the value at corner each point on the grid
        self.gridPoints = gridPoints
        #value that
        self.isoval = isoval

        if gridCoord is None:
            self.gridCoord = []
            for i in range(len(gridPoints)):
                self.gridCoord.append([])
                for j in range(len(gridPoints[i])):
                    self.gridCoord[i].append((j,len(self.gridPoints[i])-i-1))
        else:
            self.gridCoord = gridCoord

        #table that stores the 2 edges are intersected given the sign pattern.  The sign pattern goes left top, right
        #top, left bottom, right bottom and the edges are given on a 0-1 scale where 0,0 is the upper left corner
        # and 1,1 is the bottom right.
        self.isoTable = {
            "----" : [],
            "---+" : [[Edge(1,0,1,1),Edge(1,1,0,1)]],
            "--+-" : [[Edge(0,0,1,0),Edge(1,0,1,1)]],
            "-+--" : [[Edge(0,0,0,1),Edge(0,1,1,1)]],
            "+---" : [[Edge(1,0,0,0),Edge(0,0,0,1)]],
            "++--" : [[Edge(0,0,1,0),Edge(0,1,1,1)]],
            "+-+-" : [[Edge(0,0,0,1),Edge(1,0,1,1)]],
            "+--+" : [[Edge(0,0,1,0),Edge(1,0,1,1)],[Edge(0,0,0,1),Edge(0,1,1,1)]],
            "-+-+" : [[Edge(0,0,0,1),Edge(1,0,1,1)]],
            "-++-" : [[Edge(1,0,0,0),Edge(0,0,0,1)],[Edge(1,0,1,1),Edge(1,1,0,1)]],
            "--++" : [[Edge(0,0,1,0),Edge(0,1,1,1)]],
            "-+++" : [[Edge(1,0,0,0),Edge(0,0,0,1)]],
            "+-++" : [[Edge(0,0,0,1),Edge(0,1,1,1)]],
            "++-+" : [[Edge(0,0,1,0),Edge(1,0,1,1)]],
            "+++-" : [[Edge(1,0,1,1),Edge(1,1,0,1)]],
            "++++" : []
        }

        self.getSigns()
        self.getIsolines()
        self.interpLines()

    #determines the signs of each endpoint for the given isovalue.
    def getSigns(self):
        self.signs = []
        for i in range(len(self.gridPoints)):
            self.signs.append([])
            for j in range(len(self.gridPoints[i])):
                #does it need to be strictly greater or greater than equal to?
                if self.gridPoints[i][j] >= self.isoval:
                    self.signs[i].append("+")
                else:
                    self.signs[i].append("-")
        return self.signs

    #method that returns the isolines in for each grid square on a 0-1 scale.
    def getIsolines(self):
        self.isoLines = []
        #iterate over all the rows (i) and columns (j)
        for i in range(len(self.gridPoints)-1):
            self.isoLines.append([])
            for j in range(len(self.gridPoints[i])-1):
                self.isoLines[i].append([])
                #create the string of the signs to use as the key in the isoTable
                tableStr = str(self.signs[i][j]) + str(self.signs[i][j+1]) + str(self.signs[i+1][j]) + str(self.signs[i+1][j+1])

                #iterate over the pairs of edges returned by the isoTable
                # at most two lines
                for line in self.isoTable[tableStr]:
                    edgePoints = []
                    #only two edges in line
                    for edge in line:
                        endP = [0,0]
                        points = edge.getPointList()

                        #calculate the indices for the 2D array storing the values of the grid points
                        p1 = [points[0][0]+i,points[0][1]+j]
                        p2 = [points[1][0]+i,points[1][1]+j]
                        #scale the isovalue to a 0-1 scale.  A negative value means that the left end point is
                        #is greater than the right endpoint.
                        scaledIsoval = 1 / (self.gridPoints[p2[0]][p2[1]] - self.gridPoints[p1[0]][p1[1]]) * (
                            self.isoval - min(self.gridPoints[p1[0]][p1[1]], self.gridPoints[p2[0]][p2[1]]))
                        # if it was negative, subtract the absolute value of it from 1
                        #this gives the value with respect to the left coordinate instead of the right
                        if(scaledIsoval<0):
                            scaledIsoval = 1-abs(scaledIsoval)

                        #determine which coordinate shold store the scaled Isovalue
                        if p1[0]-p2[0]==0:
                            endP[0] = points[0][0]
                            endP[1] = scaledIsoval
                        else:
                            endP[0] = scaledIsoval
                            endP[1] = points[0][1]
                        edgePoints.append(endP)
                    #there could be 2 edges in a grid square (-++- or +--+) so iterate over all the points in
                    #with a skip of 2 so they are accessed in pairs corresponding to a single edge
                    for k in range(0,len(edgePoints),2):
                        self.isoLines[i][j].append(Edge(edgePoints[k][0],edgePoints[k][1],edgePoints[k+1][0],edgePoints[k+1][1]))

        return self.isoLines

    def interpLines(self):
        #list to store edges that are the final, interpolated isolines
        self.interpIsolines = []
        #iterate over each grid square to interpolate the isolines for that square
        for i in range(len(self.isoLines)):
           for j in range(len(self.isoLines[i])):
                edges = self.isoLines[i][j]
                #iterate over edges in a grid square (at most two edges)
                for k in range(len(edges)):
                    points = edges[k].getPointList()
                    # create the string of the signs to use as the key in the isoTable
                    tableStr = str(self.signs[i][j]) + str(self.signs[i][j + 1]) + str(self.signs[i + 1][j]) + str(
                        self.signs[i + 1][j + 1])
                    #get the edges corresponding to the current grid square from the isotable
                    line=self.isoTable[tableStr][k]
                    endPoints=[]
                    #at most two edges
                    for l in range(len(line)):
                        #get the end points of the edge at the current index
                        edgePoints = line[l].getPointList()
                        #get the coordinates of each endpoint
                        c1 = self.gridCoord[edgePoints[0][0]+i][edgePoints[0][1]+j]
                        c2 = self.gridCoord[edgePoints[1][0] + i][edgePoints[1][1] + j]

                        #determine which index holds the isovalue
                        if(points[l][0]==0 or points[l][0]==1):
                            scaledIsovalue = points[l][1]
                        else:
                            scaledIsovalue = points[l][0]
                        #append the interpolated enpoint to the list of current endpoints
                        endPoints.append(self.interpolatePoint(scaledIsovalue,c1,c2))

                    #add the interpolated endpoint to the list of edges (does not have the 2D array structure like the
                    #grid points and isolines)
                    self.interpIsolines.append(Edge(endPoints[0][0],endPoints[0][1],endPoints[1][0],endPoints[1][1]))



    def interpolatePoint(self,scaledIsoval,c1,c2):
        #determine whether x or y coordinate is the same for both coordinates
        if c1[0] - c2[0] == 0:
            diff = c2[1]-c1[1]
            #scale the scaled isovalue to fit on the scale of the grid square
            interpIsoval = c1[1] + (scaledIsoval * diff)
            #return the scaled point
            return [c1[0],interpIsoval]
        else:
            diff = c2[0]-c1[0]
            interpIsoval = c1[0] + (scaledIsoval * diff)
            return [interpIsoval,c1[1]]



def readFile(filename):
    read = csv.reader(open(filename, 'rt'))
    points = []
    headers = next(read)
    for row in read:
        rowDat = []
        # tokens = row.split(',')
        for i in range(0, len(row)):
            try:
                rowDat.append(float(row[i]))
            except:
                print("invalid data type - must be numeric")
                exit(0)
        points.append(rowDat)
    return points

def readGeneralFile(filename):
    read = csv.reader(open(filename, 'rt'))

    points = []
    headers = next(read)
    for row in read:
        rowDat = []
        # tokens = row.split(',')
        for i in range(0, len(row)):
            rowDat.append(row[i])
        points.append(rowDat)
    return points

def plotIsolines(plot,vVector,gridCoord):

    maxval = max(vVector)
    minval = min(vVector)
    gridPoints = []
    squareMins = []
    for i in range(len(gridCoord)):
        gridPoints.append([])
        for j in range(len(gridCoord[0])):
            gridPoints[i].append(vVector[len(gridCoord[0]) * i + j])

    for i in range(len(gridCoord) - 1):
        squareMins.append([])
        for j in range(len(gridCoord[i]) - 1):
            squareMins[i].append(-1)
    minX = gridCoord[0][0][0]
    maxY = gridCoord[0][0][1]

    numlines = 10
    for i in range(numlines):
        print(i)
        isoval = (maxval - minval) * i / float(numlines) + minval
        msq = MarchingSqaures(gridPoints, isoval, gridCoord)
        edges = msq.interpIsolines
        for edge in edges:
            p1 = edge.getP1()
            p2 = edge.getP2()
            xpoints = [p1[0], p2[0]]
            ypoints = [p1[1], p2[1]]
            curve1 = pg.PlotDataItem(xpoints, ypoints)


            corners = []

            j = 0
            k = 0
            x2 = minX
            while (x2 < p1[0] or x2 < p2[0]):
                j += 1
                x2 = gridCoord[0][j][0]

            x1 = gridCoord[0][j - 1][0]

            y2 = maxY
            while (y2 > p1[1] or y2 > p2[1]):
                k += 1
                y2 = gridCoord[k][0][1]

            y1 = gridCoord[k - 1][0][1]

            corners.append([x2, y2])
            corners.append([x2, y1])
            corners.append([x1, y2])
            corners.append([x1, y1])

            xPoints2 = []
            yPoints2 = []
            points2 = []

            if squareMins[k - 1][j - 1] == -1:
                squareMins[k - 1][j - 1] = i

            for l in range(2):
                for m in range(2):
                    val = gridPoints[k - m][j - l]
                    if val >= isoval:
                        p = corners[2 * l + m]
                        points2.append([p[0], p[1]])

            points2.sort(key=lambda x: x[0], reverse=True)
            if len(points2) > 2 and points2[2][0] != points2[1][0] and points2[1][1] != points2[2][1]:
                p = points2[2]
                points2.remove(p)
                points2.insert(0, p)
            elif len(points2) > 2 and points2[0][0] != points2[1][0] and points2[0][1] != points2[1][1]:
                p = points2[0]
                points2.remove(p)
                points2.append(p)
            if len(points2) == 1:
                points2.insert(0, p1)
                points2.append(p2)
            else:
                if p1[0] == points2[0][0] or p1[1] == points2[0][1]:
                    points2.insert(0, p1)
                    points2.append(p2)
                else:
                    points2.append(p1)
                    points2.insert(0, p2)
            for p in points2:
                xPoints2.append(p[0])
                yPoints2.append(p[1])

            if squareMins[k - 1][j - 1] == i or i == 1:
                points3 = []
                for p in corners:
                    if p not in points2:
                        points3.append(p)

                if len(points3) > 2 and points3[2][0] != points3[1][0] and points3[1][1] != points3[2][1]:
                    p = points3[2]
                    points3.remove(p)
                    points3.insert(0, p)
                elif len(points3) > 2 and points3[0][0] != points3[1][0] and points3[0][1] != points3[1][1]:
                    p = points3[0]
                    points3.remove(p)
                    points3.append(p)
                elif len(points3) == 2:
                    if (points3[0][0] == p1[0] and points3[0][1] == p2[1]) or (
                            points3[0][0] == p2[0] and points3[0][1] == p1[1]):
                        points3.remove((points3[1]))
                    elif (points3[1][0] == p1[0] and points3[1][1] == p2[1]) or (
                            points3[1][0] == p2[0] and points3[1][1] == p1[1]):
                        points3.remove(points3[0])

                if len(points3) == 1:
                    points3.insert(0, p1)
                    points3.append(p2)
                else:
                    if p1[0] == points3[0][0] or p1[1] == points3[0][1]:
                        points3.insert(0, p1)
                        points3.append(p2)
                    else:
                        points3.append(p1)
                        points3.insert(0, p2)
                xPoints3 = []
                yPoints3 = []
                for p in points3:
                    xPoints3.append(p[0])
                    yPoints3.append(p[1])
                curve2 = pg.PlotDataItem(xPoints3, yPoints3)

                fill = pg.FillBetweenItem(curve1, curve2, brush=pg.mkBrush((102 * (numlines - (i - 1)) / float(
                    numlines) + 153, 102 * (numlines - (i - 1)) / float(numlines) + 153,
                                                                            102 * (numlines - (i - 1)) / float(
                                                                                numlines) + 153)))
                fill.setPen(pg.mkPen((102 * (numlines - (i - 1)) / float(numlines) + 153,
                                      102 * (numlines - (i - 1)) / float(numlines) + 153,
                                      102 * (numlines - (i - 1)) / float(numlines) + 153)))

                plot.addItem(fill)

            curve2 = pg.PlotDataItem(xPoints2, yPoints2)
            fill = pg.FillBetweenItem(curve1, curve2, pg.mkBrush((102 * (numlines - i) / float(numlines) + 153,
                                                                  102 * (numlines - i) / float(numlines) + 153,
                                                                  102 * (numlines - i) / float(numlines) + 153)))
            fill.setPen(pg.mkPen((102 * (numlines - i) / float(numlines) + 153,
                                  102 * (numlines - i) / float(numlines) + 153,
                                  102 * (numlines - i) / float(numlines) + 153)))
            plot.addItem(fill)
            plot.adjustSize()


            pen = pg.mkPen(color=(0, 0, 0), width=1)

            plot.plot(xpoints, ypoints, pen=pen)
            pen = pg.mkPen(color=(0, 0, 0), width=1)


        for k in range(len(squareMins)):
            for j in range(len(squareMins[k])):
                if squareMins[k][j] == -1:
                    if i == 0 or gridPoints[k][j] > isoval and gridPoints[k + 1][j] > isoval and gridPoints[k][
                                j + 1] > isoval and gridPoints[k + 1][j + 1] > isoval:
                        isoval2 = (maxval - minval) * (i + 1) / float(numlines) + minval
                        if i == 9 or gridPoints[k][j] < isoval2 and gridPoints[k + 1][j] < isoval2 and \
                                        gridPoints[k][j + 1] < isoval2 and gridPoints[k + 1][j + 1] < isoval2:
                            curve1 = pg.PlotDataItem([gridCoord[k][j][0], gridCoord[k + 1][j][0],
                                                      gridCoord[k + 1][j + 1][0], gridCoord[k][j + 1][0]],
                                                     [gridCoord[k][j][1], gridCoord[k + 1][j][1],
                                                      gridCoord[k + 1][j + 1][1], gridCoord[k][j + 1][1]])
                            curve2 = pg.PlotDataItem([gridCoord[k][j + 1][0], gridCoord[k][j][0]],
                                                     [gridCoord[k][j + 1][1], gridCoord[k][j][1]])
                            fill = pg.FillBetweenItem(curve1, curve2, pg.mkBrush((102 * (numlines - i) / float(
                                numlines) + 153, 102 * (numlines - i) / float(numlines) + 153,
                                                                                  102 * (numlines - i) / float(
                                                                                      numlines) + 153)))
                            fill.setPen(pg.mkPen((102 * (numlines - i) / float(numlines) + 153,
                                                  102 * (numlines - i) / float(numlines) + 153,
                                                  102 * (numlines - i) / float(numlines) + 153)))
                            plot.addItem(fill)


def plot(classes,colors,gridcoord,points,resultVect):
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    qApp = QtGui.QApplication([])
    mw = QtGui.QMainWindow()
    view = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(view)
    mw.show()
    plot = view.addPlot()


    plot.pen = pg.mkPen(0, 0, 0)
    plot.brush = pg.mkBrush(0, 0, 0)
    if len(resultVect) > 2 * len(points):
        resultVect = resultVect[:2 * len(points)]

    g = Grid(points, resultVect, gridCoord)

    vVector = g.calcGridPoints()
    plotIsolines(plot,vVector,gridCoord)
    points = np.array(points)

    if colors is not  None:
        if classes is not None:
            l = pg.LegendItem((100, 60), offset=(70, 30))  # args are (size, offset)
            l.setParentItem(plot)
            uniqClass = set(pd.classes)
            uniqClass = list(uniqClass)
            uniqClass.sort()
            for i, name in enumerate(uniqClass):
                pts = [i for i, x in enumerate(classes) if x == name]
                print(len(pts))

                brush = pg.mkBrush(pd.colors[i][0], colors[i][1], colors[i][2])
                scatter = pg.ScatterPlotItem(points[pts, 0], points[pts, 1], brush=brush)
                plot.addItem(scatter)
                l.addItem(scatter, name)
        else:
            pltPoints = []
            for i in range(len(points)):
                print(i)
                brush = pg.mkBrush(colors[i][0], colors[i][1],
                                colors[i][2])
                pltPoints.append({'pos': (points[i][0], points[i][1]), 'brush': brush})
            s = pg.ScatterPlotItem()
            s.addPoints(pltPoints)

            plot.addItem(s)
    else:
        plot.addItem(pg.ScatterPlotItem(points[:, 0], points[:, 1], pen=(0, 255, 0)))

    temp = np.array(gridCoord[0]).T
    plot.setXRange(min(temp[0]), max(temp[0]))
    temp = np.array(np.array(gridCoord)[:, 0]).T
    plot.setYRange(min(temp[1]), max(temp[1]))
    sys.exit(qApp.exec_())


def generateGrid(points):
    gridSize = 10
    points = np.array(points)
    xmax = max(points[:, 0])
    xmin = min(points[:, 0])
    ymax = max(points[:, 1])
    ymin = min(points[:, 1])

    gridCoord = []
    if (xmin == xmax):
        xmax += 1
        xmin -= 1
    if (ymin == ymax):
        ymin -= 1
        ymax += 1
    if ymax > xmax:
        xmax = ymax
    else:
        ymax = xmax
    if ymin < xmin:
        xmin = ymin
    else:
        ymin = xmin
    yrange = ymax - ymin
    xrange = xmax - xmin

    xstep = float(xrange) / (gridSize - 1)
    ystep = float(yrange) / (gridSize - 1)

    for i in range(gridSize + 1):
        gridCoord.append([])
        for j in range(gridSize + 1):
            gridCoord[i].append([(xmin - xstep / 2.0) + xstep * j, (ymax + ystep / 2.0) - ystep * i])
    return gridCoord


def calcColors(vals):
        try:
            vals = [float(i) for i in vals]
            colors = [0] * len(self.points)
            ran = max(vals) - min(vals)

            minC = min(vals)
            if (minC < 0):
                for i in range(len(vals)):
                    vals[i] += abs(minC)
            else:
                for i in range(len(vals)):
                    vals[i] -= abs(minC)
            for i in range(len(self.points)):
                med = .5 * ran + min(vals)
                if vals[i] < med:
                    colors[i] = [255, 255 * vals[i] / med, 0]

                else:
                    colors[i] = [0, 255 * (1 - (vals[i] - med) / med), 255 * (vals[i] - med) / med]
            return colors
        except:
            return None

def generateClassColors(self,numClasses):
        cols = [[227, 26, 28],[178, 223, 138],[66, 206, 227],[255, 127, 0],[31, 120, 180],  [51, 160, 44], [251, 154, 153],
                [253, 191, 111],  [202, 178, 214], [106, 61, 154]]
        cols = np.array(cols)

        if numClasses<=10:
            return cols[:numClasses]
        else:
            return None

if __name__=="__main__":
    classes = False
    numArgs = len(sys.argv)
    if (numArgs >= 2):
        inputFile = sys.argv[1]
        if numArgs >= 3:
            classes = True
            classFile = sys.argv[2]
        if numArgs==4:
            classCol = int(sys.argv[3])
        else:
            classCol = 0
    else:
        print("Invalid number of arguments")
        print("DimReaderPlot dimReaderOutput.csv fileWithClasses.csv(optional) classColumn(optional)")
        exit(0)

    data = np.array(readFile(inputFile))
    points = data[:,0:2]
    colors =None
    if classes:
        classData = np.array(readGeneralFile(classFile))
        classes = classData[:,classCol]
        m = len(set(classes))
        if m <= 10:
            colors= generateClassColors(m)
        else:
            colors = calcColors(classes)

    resultVect = []
    for i in range(len(data)):
        resultVect.append(data[i][2])
        resultVect.append(data[i][3])

    gridCoord = generateGrid(points)
    plot(classes,colors,gridCoord,points,resultVect)
