def getBlobSize(blob):
    blob_size = []
    blob_size.append(blob[1].stop - blob[1].start)
    blob_size.append(blob[0].stop - blob[0].start)
    return blob_size

def getBlobPosition(blob):
    blob_center = []
    blob_center.append((blob[1].stop + blob[1].start) / 2)
    blob_center.append((blob[0].stop + blob[0].start) / 2)
    return blob_center
