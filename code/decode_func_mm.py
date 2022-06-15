def decode(encoded):
  '''
  The linestring resulting from Valhalla Docker map-matching procedure 
  is in an encoded polyline format: this function decodes it and returns
  a list of lists of coordinates (lon, lat).
  '''
  inv = 1.0 / 1e6
  decoded = []
  previous = [0,0]
  i = 0
  #for each byte
  while i < len(encoded):
    #for each coord (lat, lon)
    ll = [0,0]
    for j in [0, 1]:
      shift = 0
      byte = 0x20
      #keep decoding bytes until you have this coord
      while byte >= 0x20:
        byte = ord(encoded[i]) - 63
        i += 1
        ll[j] |= (byte & 0x1f) << shift
        shift += 5
      #get the final value adding the previous offset and remember it for the next
      ll[j] = previous[j] + (~(ll[j] >> 1) if ll[j] & 1 else (ll[j] >> 1))
      previous[j] = ll[j]
    # scale by the precision and chop off long coords 
    # flip coordinates so that it returns them in the standard order, namely [lon, lat]
    decoded.append([float('%.6f' % (ll[1] * inv)), float('%.6f' % (ll[0] * inv))])
    # decoded.append([float('%.6f' % (ll[0] * inv)), float('%.6f' % (ll[1] * inv))]) # lat, lon
  return decoded