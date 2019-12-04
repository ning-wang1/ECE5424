Two folders:

Attack:
	Step 0. Install Scikit-learn for python.

	Step 1. Run trainNN.py to create a target neural network classifier. Tune model parameters as needed. The "oracle" is the output that will be used as the target model.

	Step 2. Run trainSbst_NN_h1.py to train a substitute model. Parameters matter! Please help yourself tuning them to get a highest subtitute score. The "sbst" is the output that will be used as the substitute model.

	Step 3. Run testSbst.py to craft and evaluate adversarial samples. FGS runs quite fast but OPT-L2 doesn't. As a reference, it took about 8 hours to run for 2,000 test images. You need to manually comment/uncomment corresponding code blocks to enable FGS or OPT-L2.

Defense:

	run main.py to see the success rate of adversarial attack on different models.
