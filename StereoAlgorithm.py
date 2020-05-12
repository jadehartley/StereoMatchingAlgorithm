import numpy as np
from PIL import Image
import os
import numba as nb
dirname = os.path.dirname(__file__)
variance = 16

@nb.njit
def createDisparityMap(leftImage, rightImage, occlusionCost, leftDisparityMap, rightDisparityMap):
    """
    This function takes two arrays and computes their left and right disparity maps
    To do this it creates a cost matrix and direction matrix for each row to enable
    a backwards pass to be completed

    Args:
        leftImage (np.ndarray): array representing left image
        rightImage (np.ndarray):array representing right image
        occlusionCost (float): cost of occlusion
        leftDisparityMap (np.ndarray): empty array to be filled with left disparity values
        rightDisparityMap (np.ndarray): empty array to be filled with right disparity values
    Returns:
        leftDisparityMap (np.ndarray): result for the left disparity map for the given images
        rightDisparityMap (np.ndarray):result for the right disparity map for the given images
    """

    rowCount = leftImage.shape[0]
    columnCount = leftImage.shape[1]

    for rowNo in range(rowCount):
        leftRow = leftImage[rowNo]
        rightRow = rightImage[rowNo]

        costMatrix = np.zeros((columnCount + 1, columnCount + 1))
        directionMatrix = np.ones((columnCount + 1, columnCount + 1), dtype=np.uint8)

        # Initialise the topmost row and leftmost column of the costMatrix, inclusive
        for i in range(0, columnCount + 1):
            costMatrix[i, 0] = i * occlusionCost
            costMatrix[0, i] = i * occlusionCost

        for i in range(1, columnCount + 1):
            z1 = leftRow[i - 1]
            for j in range(1, columnCount + 1):
                z2 = rightRow[j - 1]
                matchingCost = pow((z1 - z2), 2) / variance
                # Case 1 - pixels i and j match
                min1 = costMatrix[i - 1][j - 1] + matchingCost
                # Case 2 - pixel i is unmatched
                min2 = costMatrix[i - 1][j] + occlusionCost
                # Case 3 - pixel j is unmatched
                min3 = costMatrix[i][j - 1] + occlusionCost

                # The value in the cost matrix is the minimum of the above cases
                minimum = min(min1, min2, min3)
                costMatrix[i][j] = minimum

                if minimum == min3:
                    directionMatrix[i][j] = 3
                elif minimum == min1:
                    directionMatrix[i][j] = 1
                elif minimum == min2:
                    directionMatrix[i][j] = 2
                else:
                    print("Error matching minimum")
                    print(i, j)
                    print(costMatrix[i][j], int(min1), int(min2), int(min3), min1, min2, min3)

        # Perform a backwards traversal using the directionMatrix to construct optimal match
        i = columnCount #i represents the rows in our directionMatrix (left image pixels)
        j = columnCount #j represents the columns (right image pixels)

        while (i != 0) and (j != 0):
            if directionMatrix[i][j] == 1:
                # Disparity here is multiplied by 10 and added to 128
                # This makes disparity values more distinct and occlusions more obvious
                disparity = abs(i - j) * 10 + 128
                leftDisparityMap[rowNo][i] = disparity
                rightDisparityMap[rowNo][j] = disparity
                i -= 1
                j -= 1
            elif directionMatrix[i][j] == 2:
                i -= 1
            elif directionMatrix[i][j] == 3:
                j -= 1
            else:
                print("Error within direction matrix")

    # Interpolate values to map them between range [0,255]
    leftDisparityMap *= 255.0 / leftDisparityMap.max()
    rightDisparityMap *= 255.0 / rightDisparityMap.max()
    leftDisparityMap = leftDisparityMap.astype(np.uint8)
    rightDisparityMap = rightDisparityMap.astype(np.uint8)
    return leftDisparityMap, rightDisparityMap


def go(occlusionCost):
    """
    This function does the bulk of the image processing, importing images, converting them to greyscale and storing
    as arrays. It calls a function to compute their disparity maps for the given occlusion cost before outputting 
    the results
    
    Args: 
        occlusionCost (float): cost of occlusion
    Outputs:
        2 images representing the left and right disparity maps

    """
    leftImage = np.array(Image.open(os.path.join(dirname, 'randomdotleft.png')).convert('L'))
    rightArr = np.array(Image.open(os.path.join(dirname, 'randomdotright.png')).convert('L'))

    leftDisparityMap = np.zeros(leftImage.shape, dtype='float64')
    rightDisparityMap = np.zeros(leftImage.shape, dtype='float64')

    leftDisparityMap, rightDisparityMap = createDisparityMap(leftImage, rightArr, occlusionCost, leftDisparityMap, rightDisparityMap)

    Image.fromarray(leftDisparityMap).save(os.path.join(dirname, "out2/Displeft" + str(occlusionCost) + ".png"))
    Image.fromarray(rightDisparityMap).save(os.path.join(dirname, "out2/Dispright" + str(occlusionCost) + ".png"))


if __name__ == "__main__":
    for i in np.arange(0.5, 5, 0.25):
        go(round(i, 1))
        print(i)
