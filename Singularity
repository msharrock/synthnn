Bootstrap: docker
From: jcreinhold/synthnn

%environment
. /opt/conda/bin/activate synthnn

%runscript
exec python $@

%apprun train
exec nn-train $@

%apprun predict
exec nn-predict $@

%apprun fa-train
exec fa-train $@

%apprun fa-predict
exec fa-predict $@
