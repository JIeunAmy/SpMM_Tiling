#!/bin/bash

mkdir data
cd data
wget https://suitesparse-collection-website.herokuapp.com/MM/Um/2cubes_sphere.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage12.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/cant.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/consph.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/cop20k_A.tar.gz
#no facebook_combined
wget https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/filter3D.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/hood.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/JGD_Homology/m133-b3.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/mac_econ_fwd500.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/QLi/majorbasis.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/mc2depi.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Um/offshore.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Pajek/patents_main.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/FEMLAB/poisson3Da.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Bova/rma10.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Hamm/scircuit.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec1.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-BerkStan.tar.gz


find . -name '*.tar.gz' -exec tar xvf {} \;
rm *.tar.gz
cp ../conv.c .
gcc -O3 -o conv conv.c

for i in `ls -d */`
do
cd ${i}
ii=${i/\//}
mv ${ii}.mtx ${ii}.mt0
../conv ${ii}.mt0 ${ii}.mtx 
rm ${ii}.mt0
cd ..
done

cd ..

