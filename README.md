ADCP MODEL
BP & XGBOOST &ATTENTION

1. Environment
    Python environment is recommended, and the version can be selected according to your computer to ensure that torch and torch nn can run smoothly. Please refer to torch official website for version selection
   The relevant code can reference as following tutorial: 
    conda create --name py38 python =3.8
    conda init
    conda activate py38
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    Also, you need to install the necessary packages as Pandas, matplotlib...
2. Running
    We provide all the code so you can play with it as you wish. At the same time, the real car data set is provided. You can train the model you need by replacing the training data, paying attention to the dimension of the data needs to be corresponding. Of course, you can also use your own data. 
