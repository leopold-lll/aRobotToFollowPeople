# A Robot To Follow People
This repository is dedicated to the master thesis research project of **Stefano Leonardi** at the **[University of Trento](https://www.unitn.it/en)** for the **[Dolomiti Robotics](https://dolomitirobotics.it/)** group.

### The goal
The goal of this code is to allow a robot equipped with an RGB camera to detect, identify and track a person in a real-time video. Then, with the use of a LIDAR sensor follows it across time through a real environment.

### Necessary  material
The repository contains only the code but the entire project to work requires additional elements:
- The DNN (Deep Neural Network) [pre-trained models](https://drive.google.com/drive/folders/1NIsFhys1TO4IEbt0ZBErVAGFfm2TB14L?usp=sharing).
- The [pre-computed encodings](https://drive.google.com/drive/folders/17UaRxobv3ESAeaWWYVSJQzaXXCK4aq4a?usp=sharing) of the database of images of people.

And, **optionally** also:
- The [database of images](https://drive.google.com/drive/folders/1UG_BCHDNZywuIp5mIVAFgByNDbOBzKWA?usp=sharing) internally used as samples when there are no people in the field of view of the robot.
- A [set of input samples](https://drive.google.com/drive/folders/1v_NtzNaYFYeP-5k7Mzwyy3rw1KzmDXAI?usp=sharing) recorded to test the potentialities of the software.

These folders should be placed in the root location of the project.

### References
A quick and complete view of the project can be done with the use of:
- A [demo](https://drive.google.com/file/d/1s_sXa-Q7-MVhQVobPWRdU7K5oFsKyt4N/view?usp=sharing) of the working software executed on an Intel Core i5 CPU.
- The [final dissertation](https://github.com/leopold-lll/thesis_aRobotToFollowPeople/blob/master/main.pdf) containing all the details and the implemental choices that have been done.
The dissertation is titled: *"Integration of multiple deep learning algorithms for real-time tracking of a person in complex scenarios"*.
- The powerpoint [presentation](https://drive.google.com/file/d/1jJQ9YGHTVK5UrLhepHpZLqO4kwQCIXDb/view?usp=sharing) of the overall project.
- The [recorded oral discussion](https://drive.google.com/file/d/1vLVMeXBxDt49J976ht0A8TKUEt3Bqjcf/view?usp=sharing) of the presentation.
