import numpy as np 
import matplotlib.pyplot as plt 
from qiskit import IBMQ, assemble, transpile 
from qiskit import * 
from statsmodels.graphics.tsaplots import plot_acf 
from qiskit import BasicAer
shots = 65536

N_QUBITS = 20 
G = 0.98 
T = 40

class DiscreteTimeCrystal:

	def __init__(self, n_qubits: int) -> None:
		self.n_qubits = n_qubits
		self.backend = BasicAer.get_backend('qasm_simulator')
	def random_bitstring_circuit(self) -> QuantumCircuit:
		"""
		Args:
			n_qubits: number of qubits in the circuit
		Returns:
			QuantumCircuit: object that creates a random bitstring from the ground state
		"""
		qc = QuantumCircuit(self.n_qubits)
		random_state = np.random.randint(2, size=self.n_qubits)
		for i in range(self.n_qubits):
			if random_state[i]:
				qc.x(i)
		return qc

	def floquet_circuit(self, n_qubits: int, g: float) -> QuantumCircuit:
		"""
		Args:
			n_qubits: number of qubits in the floquet_circuit
			g: parameter in range [0.5, 1] controlling the magnitude of x-rotation
		Returns:
			QuantumCircuit: implementation of the Floquet unitary circuit U_f described 
				in https://arxiv.org/pdf/2107.13571.pdf
		"""

		qc = QuantumCircuit(n_qubits)

		# X rotation by (pi * g)
		for i in range(n_qubits):
			qc.rx(np.pi * g, i)

		# Ising interaction (only coupling adjacent spins)
		for i in range(0, n_qubits-1, 2):
			phi = np.random.uniform(low=0.5, high=1.5)
			theta = np.pi * phi / 2
			qc.rzz(-theta, i, i+1)
		for i in range(1, n_qubits-1, 2):
			phi = np.random.uniform(low=0.5, high=1.5)
			theta = np.pi * phi / 2
			qc.rzz(-theta, i, i+1)

		# Longitudinal fields for disorder
		for i in range(n_qubits):
			h = np.random.uniform(low=-1, high=1)
			qc.rz(np.pi * h, i)

		return qc

	def mean_polarization(self, counts: dict, q_index: int) -> float:
		"""
		Args:
			counts: dictionary of measurement results and corresponding counts
			q_index: index of qubit in question
		Returns:
			float: the mean polarization, in [-1, 1], of the qubit at q_index, as given
				by the counts dictionary
		"""
		exp, num_shots = 0, 0
		for bitstring in counts.keys():
			val = 1 if int(bitstring[self.n_qubits-q_index-1]) else -1
			exp += val * counts[bitstring]
			num_shots += counts[bitstring]
		return exp / num_shots

	def acf(self, series):
	    n = len(series)
	    data = np.asarray(series)
	    mean = np.mean(data)
	    c0 = np.sum((data - mean) ** 2) / float(n)

	    def r(h):
	        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
	        return round(acf_lag, 3)
	    x = np.arange(n) # Avoiding lag 0 calculation
	    acf_coeffs = list(map(r, x))
	    return acf_coeffs

	def simulate(self, initial_state: QuantumCircuit, T: float, g: float, plot=False) -> None:
		exp_arr = []
		floq_qc = self.floquet_circuit(self.n_qubits, g)
		for t in range(1, T):
			qc = QuantumCircuit(self.n_qubits)
			qc = qc.compose(initial_state)
			for i in range(t):
				qc = qc.compose(floq_qc)
			qc.measure_all()
			transpiled = transpile(qc, backend=self.backend)
			job = self.backend.run(assemble(transpiled, backend=self.backend, shots=shots, memory=True))
			print("Jobs submitted to %s. Job ID is %s.", self.backend, job.job_id())
#			result = execute(circuit, backend=self.backend)
			counts=job.result().get_counts()
#			counts=job.get_counts(circuit)
#			job = self.backend.run(transpiled)
#			counts = BasicAer.BasicAerJob.result(result(transpiled, backend=self.backend, shots=shots, memory=True))
			exp = self.mean_polarization(counts, 11)
			exp_arr.append(exp)
		if plot:
			plt.plot(range(1, T), exp_arr, 'ms-')
			autocorr = self.acf(exp_arr)
			print(autocorr)
			plt.plot(range(1, T), autocorr, 'bs-')
			plt.show()
		return exp_arr



dtc = DiscreteTimeCrystal(n_qubits=N_QUBITS)
exp = []
ac = np.zeros(shape=(T-1))
for j in range(36):
	print(j)
	initial_state = dtc.random_bitstring_circuit()
	q11_z_exp = dtc.simulate(initial_state=initial_state, T=T, g=G, plot=False)
	q11_z_ac = dtc.acf(q11_z_exp)
	ac += np.array(q11_z_ac)
print(ac)
ac = ac / 36
plt.plot(range(1, T), ac, 'bs-')
plt.show()



