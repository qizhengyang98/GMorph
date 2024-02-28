cd benchmark_scripts ;

# ------ random sampling ------
# accuracy drop < 0
./b1_random_t0.sh ;
# accuracy drop < 0.01
./b1_random_t001.sh ;
# accuracy drop < 0.02
./b1_random_t002.sh ;

# ------ GMorph ------
# accuracy drop < 0
./b1_SA_t0.sh ;
# accuracy drop < 0.01
./b1_SA_t001.sh ;
# accuracy drop < 0.02
./b1_SA_t002.sh ;

# ------ GMorph w P ------
# accuracy drop < 0
./b1_LC_t0.sh ;
# accuracy drop < 0.01
./b1_LC_t001.sh ;
# accuracy drop < 0.02
./b1_LC_t002.sh ;

# ------ GMorph w P+R ------
# accuracy drop < 0
./b1_LC_t0_R.sh ;
# accuracy drop < 0.01
./b1_LC_t001_R.sh ;
# accuracy drop < 0.02
./b1_LC_t002_R.sh ;