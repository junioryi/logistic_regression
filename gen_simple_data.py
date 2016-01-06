import numpy as np

data = np.random.rand(10, 10)
label = []
for d in data:
	for dummy in range(5):
		d[ np.random.randint(10) ] = 0
	
	pos = sum(d[:5])
	neg = sum(d[5:])
	
	label.append( 1 if pos > neg else 0 )

# Output sparse data format
with open("simple_data", 'w') as f:
	for idx, d in enumerate(data):
		f.write( str(label[idx]) )
		for feature_idx, value in enumerate(d):
			if value != 0:
				f.write( ' ' + str(feature_idx) + ':' + str(value) )
		f.write("\n")



		

print data
print label

