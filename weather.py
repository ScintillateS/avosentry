import requests
import json
import math
url = 'https://api.windy.com/api/point-forecast/v2'
latitude = 49.809
longitude = 16.787
myobj = {"lat": latitude,
    "lon": longitude,
    "model": "gfs",
    "parameters": ["wind"],
    "levels": ["surface"],
    "key": "bzKT7ODJ9WUl8LvNJuvc9EieeW329x91"}
x = requests.post(url, json = myobj)
y = json.loads(x.text)
latestxwind = y['wind_u-surface'][-1] #latest wind data point from the x direction (west to east)
latestywind = y['wind_v-surface'][-1] #latest wind data point from the y direction (south to north)

arrowendpointx = longitude + (latestxwind*50)
arrowendpointy =  latitude + (latestywind*50)
windslope = (arrowendpointy-latitude)/(arrowendpointx-longitude)
thicknessslope = (-1)/(windslope)

cornerrectx = longitude + (3/math.sqrt(1+(thicknessslope**2)))
cornerrecty = latitude + ((3*thicknessslope)/math.sqrt(1+(thicknessslope**2)))
secondcornerrectx = longitude - (3/math.sqrt(1+(thicknessslope**2)))
secondcornerrecty = latitude - ((3*thicknessslope)/math.sqrt(1+(thicknessslope**2)))

finalcornerx = secondcornerrectx + (latestxwind*50)
finalcornery = secondcornerrecty + (latestywind*50)


print(arrowendpointx, arrowendpointy)

print(longitude, latitude)

print(cornerrectx, cornerrecty)

print(secondcornerrectx, secondcornerrecty)

print (finalcornerx, finalcornery)

print(windslope, thicknessslope)

#FOR RECTANGLE USE cornerrectx, cornerrecty AND finalcornerx, finalcornery