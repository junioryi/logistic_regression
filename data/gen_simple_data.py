import numpy as np

num_data = 6
num_feature = 6
data = np.random.rand( num_data, num_feature)
label = []
for d in data:
	for dummy in range(num_feature/2):
		d[ np.random.randint(num_feature) ] = 0
	
	pos = sum(d[:num_feature/2])
	neg = sum(d[num_feature/2:])
	
	label.append( 1 if pos > neg else 0 )

# Output sparse data format
with open("simple_data", 'w') as f:
	for idx, d in enumerate(data):
		f.write( str(label[idx]) )
		for feature_idx, value in enumerate(d):
			if value != 0:
				f.write( ' ' + str(feature_idx+1) + ':' + str(value) )
		f.write("\n")


