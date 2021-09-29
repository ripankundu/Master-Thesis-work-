# Master-Thesis-work-
Title- INVESTIGATING EVENTS AND ANOMALY DETECTION FORCYBER-PHYSICAL POWER SYSTEM USING ARTIFICIALINTELLIGENCE

#Abstract
Power system security is a mission-critical factor in modern power generation, transmission and distribution
systems. Nowadays any modern power system is connected with numerous cyber components essentially
forming cyber-physical systems (CPS). As one of the largest and most complex systems on earth, power grid
(PG) operation and control have stepped forward as a compound analysis on both physical and cyber layers
which makes it vulnerable to assaults from economic and security considerations. Machine learning (ML) is a
form of Artificial intelligence (AI) that teaches computers to think in a similar way to how humans do: Learning
and improving upon past experiences. It works by exploring data and identifying patterns and involves minimal
human intervention. Machine learning models can catch complex patterns that would have been overlooked
during human analysis. Machine learning techniques can be mainly classified into three types: supervised,
unsupervised, and reinforcement learning.
Machine learning (ML) algorithms have been widely applied in power grid functions for control and
monitoring purposes. The major reasons for ML’s thriving in the power systems are the following:
Machine learning includes a large family of algorithms for solving different problems ranging from
classification, regression, prediction to stochastic optimization, and so on. It is very straightforward to formulate
a problem under the frame of an ML algorithm. Most of the ML methods are data-driven and modern power
systems are equipped with sufficient monitoring devices to produce the data to support the implementation of
these ML algorithms.
Given a large number of protection devices, energy storage, and distributed generators installed in the current
system, a failure or a scheduled/unscheduled action from them can cause an immediate substantial change and
induce different levels of reliability and stability issues to the power grids. Also, the increasing penetration of
renewable energy brings severe uncertainties into the grid. With an increasing number of extreme events, grid
components, and complexity, more alarms are being observed in the power grid control centers. In a power
system, events refer to abnormal operations which, present themselves as oscillations, transients, short
duration variations, long-duration variations, and waveform distortion in voltage, current, and frequency values
that are usually caused by bus faults, line tripping, switching actions, topology changes, controllers, load
behavior and source dynamics. If an abnormal condition happens in a power system, for example, an overload
due to sudden load reconnection, a generator outage or a transmission line tripping, the frequency and voltage
instabilities, etc., the abnormality must be quickly addressed. If these abnormal conditions are allowed to
continue and not addressed properly in time, it would experience cascading events that might lead to a
blackout. So, operators in the control center need to monitor and analyze these alarms to take suitable control
actions, if needed, to ensure the system's reliability, stability, security, and resiliency. Hence, an early and
accurate detection, location, and classification of events can increase safety and reliability and reduce
downtime and interruption time.
Modern cyber-physical power systems generally depend on many integrated sensors including phasor
measurement units (PMU's) generating huge amounts of data every moment. Phasor Measurement Units
(PMU's) allow the online analysis of dynamic events, including fault events and oscillation events, which is not
achievable with traditional supervisory control and data acquisition measurements. The complex infrastructure
to deliver high-resolution timestamped data makes the synchrophasor measurements prone to bad data, such
as missing data or outliers. Synchrophasor data with its high reporting rate tend to capture the power system
events occurring in the system, called event data. Events and Anomaly detection have become one of the
challenging tasks for critical infrastructure systems since detecting anomalous events are critical. especially for
the cyber-physical power systems. Event data can be faults in the system, switching operations, load changes,
generator drop, and other such power system events. If fed with low-quality or erroneous data, applications
might produce a result that can be misleading. Therefore, it is pertinent to detect and classify an anomaly into
bad data or event data to aid and improve the performance of synchrophasor-based applications. Anomaly
detection is known as outlier detection is the process of discovering patterns in data that do not conform to
expected behavior. Anomalies are patterns in data that do not follow the expected behavior and they are rarely
encountered. These large chunks of data make the problem of detecting anomalies and classifying them as bad
data or event data a very challenging task. To detect data anomalies, several model-based methods have been
proposed, But, the model-based approaches heavily depend on the knowledge of system topology and model
parameters. Their detection performances can deteriorate when the system topology and the model
parameters are unknown. Therefore, data-driven approaches are proposed to detect the anomalies in the
synchrophasor data, eliminating the reliance on the system topology and model parameters. As we know that
data-driven models adopting Machine learning (ML) techniques are more like a “black-box”, with no explicit
analytical description of defining a relationship between input and output. It is based on the anomalous feature
extracted from a large set of experimental input data set and to form a function relating inputs and outputs.
Additionally, pre-trained models can detect events/anomalies within a short period of time. In some cases, pretrained
models can provide early warning to the system operators for maintenance. Different machine learning
approaches have been introduced to efficiently detect events and anomalies for large infrastructures like cyberphysical
power systems including Support vector machine (SVM), K-means clustering, Random Forest,
Convolution Neural Network (CNN), Generative Adversarial Network, Recurrent Neural Network, etc. Their
design and performance vary and depend on different aspects. This study aims to investigate the event and
anomaly detection and classification techniques using state-of-the-art machine learning algorithms on cyberphysical
power systems' measurements and monitoring data (e.g., voltage, angle, real and reactive power,
electricity consumption data, etc.). This study will identify the efficient and robust machine learning technique
for event and anomaly detection and classification on cyber-physical power systems. The main focus of this
thesis work.
• Real-time anomaly detection using unsupervised machine learning with high accuracy and using limited
memory bound.
• For the anomaly detection Semi-supervised recurrent LSTM autoencoder is applied to only learn the
normalities of the power system measurement where the deep network structure based on long short
term memory (LSTM) can automatically fit on the high-dimensional and nonlinear measurements
without prior model knowledge.
• Classification of anomalous data into bad-data and power system events using Machine learning (ML)
technique
