from collections import Counter

class Analyzer:
	def __init__(self, array):
		self.a = array
		self.len = len(array)
		# self.count = None
		# self.dist = None
		self._run()
	def _run(self):
		self.count = Counter(self.a)
		self.dist = {k: v/self.len for k, v in self.count.items()}
		print(self.count)
		print(self.dist)