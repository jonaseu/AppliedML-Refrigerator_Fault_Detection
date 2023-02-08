# AppliedML-Refrigerator_Fault_Detection
Work for discipline INE410146 - Applied Machine Learning for Graduate Program in Computer Science (PPGCC) - Federal University of Santa Catarina

# Objective: 
Design and build a machine learning pipeline to classify faults of a refrigerator

# Motivation:
No fault found (NFF) accounts for about 30-70\% of the returned faulty products. Also, the investigation into the diagnosis of household anomalous appliances has not been decently taken into consideration \cite{Hosseini2020}. In the end, that means that home appliance manufacturing companies need to spend a large amount of capital caused by an imprecise or incorrect fault detection done by the maintenance engineer or by a misperception of the user. Such a fault can be identified by classifying the multivariate time series (MTS).The present work shows a machine learning pipeline that can receive an MTS from a refrigerator and classify that product as not having fault or having some fault on which the model was trained on.

pip freeze > requirements.lock.txt
