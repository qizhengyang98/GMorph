gdown --folder https://drive.google.com/drive/folders/1Dtvd5eIDeDiseCAwCrj3_wrqjWsy3bq3?usp=sharing ;

mkdir test_metamorph/scene/pre_models ;
mv GMorph_AE/salientNet.model test_metamorph/scene/pre_models/ ;
mv GMorph_AE/salientNet_vgg16.model test_metamorph/scene/pre_models/ ;
mv GMorph_AE/objectNet.model test_metamorph/scene/pre_models/ ;

mkdir test_metamorph/face/pre_models ;
mv GMorph_AE/EmotionNet.model test_metamorph/face/pre_models/ ;
mv GMorph_AE/EmotionNet_vgg13.model test_metamorph/face/pre_models/ ;
mv GMorph_AE/ageNet.model test_metamorph/face/pre_models/ ;
mv GMorph_AE/genderNet.model test_metamorph/face/pre_models/ ;
mv GMorph_AE/genderNet_vgg11.model test_metamorph/face/pre_models/ ;

mv GMorph_AE/age_gender.gz metamorph/data/ ;
mv GMorph_AE/toy_vgg13.pt metamorph/model/ ;

mv GMorph_AE/cola.zip test_metamorph/transformer_model/ ;
mv GMorph_AE/sst2.zip test_metamorph/transformer_model/ ;
mv GMorph_AE/salient.zip test_metamorph/transformer_model/ ;
mv GMorph_AE/multiclass.zip test_metamorph/transformer_model/ ;
unzip test_metamorph/transformer_model/cola.zip -d test_metamorph/transformer_model/ ;
rm test_metamorph/transformer_model/cola.zip;
unzip test_metamorph/transformer_model/sst2.zip -d test_metamorph/transformer_model/ ;
rm test_metamorph/transformer_model/sst2.zip ;
unzip test_metamorph/transformer_model/salient.zip -d test_metamorph/transformer_model/ ;
rm test_metamorph/transformer_model/salient.zip ;
unzip test_metamorph/transformer_model/multiclass.zip -d test_metamorph/transformer_model/ ;
rm test_metamorph/transformer_model/multiclass.zip ;

mv GMorph_AE/datasets.zip ./ ;
unzip datasets.zip ;
rm datasets.zip ;

rm -rf GMorph_AE ;