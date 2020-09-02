import os
f = os.popen("/opt/vc/bin/vcgencmd measure_temp")
temp = f.read()
print(type(temp))