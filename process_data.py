BAREA = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

with open('sfo_data.csv', 'r') as f:
	with open('sfo_data_clean.csv', 'w') as g:
		columns = f.readline().strip().split(',')
		g.write(','.join(columns) + '\n')

		for row in f:
			r = row.strip().split(',')
			r[7] = str(BAREA[r[7]])
			for i in range(len(r)):
				if not r[i].isdigit() or not r[i]:
					r[i] = '-1'

			g.write(','.join(r) + '\n')


