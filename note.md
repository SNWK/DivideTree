### Data：

###### alliso-sorted.txt

https://github.com/akirmse/mountains

The isolation file has one peak per line, in this format:

latitude,longitude,elevation in feet,ILP latitude,ILP longitude,isolation in km

where ILP means isolation limit point.

A zip file with our isolation results for the world is [here](https://drive.google.com/file/d/0B3icWNhBosDXRm1pak56blp1RGc/view?usp=sharing).



###### prominence-p100.txt

https://github.com/akirmse/mountains

The prominence file has one peak per line, in this format:

latitude,longitude,elevation in feet,key saddle latitude,key saddle longitude,prominence in feet

A zip file with our prominence results for the world is [here](https://drive.google.com/file/d/0B3icWNhBosDXZmlEWldSLWVGOE0/view?usp=sharing).



两个文件 latitude longitude 有细微差距：比如 -9.1217,-77.6042 和 -9.1208,-77.6050 

所以在match的时候有一个距离threshold