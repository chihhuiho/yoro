#Please refer to https://github.com/dandelin/ViLT/blob/master/DATA.md for more details

DIR="datasetsv2"
if [ ! -d $DIR ]; then
    echo "creating $DIR"
    mkdir $DIR
fi
DIR=$DIR"/raw"
if [ ! -d $DIR ]; then
    echo "creating $DIR"
    mkdir $DIR
fi

# COCO datasets
COCODIR=$DIR"/COCO"
if [ ! -d $COCODIR ]; then
    echo "creating $COCODIR"
    mkdir $COCODIR
fi
if [ ! -f $COCODIR/train2014.zip ]; then
    echo "Downloading COCO train 2014"
    wget http://images.cocodataset.org/zips/train2014.zip -P $COCODIR
    unzip $COCODIR/train2014.zip -d $COCODIR
fi
if [ ! -f $COCODIR/val2014.zip ]; then
    echo "Downloading COCO val 2014"
    wget http://images.cocodataset.org/zips/val2014.zip -P $COCODIR
    unzip $COCODIR/val2014.zip -d $COCODIR
fi
if [ ! -f $COCODIR/caption_datasets.zip ]; then
    echo "Downloading COCO karpathy split"
    wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip -P $COCODIR
    unzip $COCODIR/caption_datasets.zip -d $COCODIR
    mkdir $COCODIR/karpathy
    ln -s $PWD/$COCODIR/dataset_coco.json $COCODIR/karpathy/dataset_coco.json
fi

# F30K dataset
F30KDIR=$DIR"/F30K"
if [ ! -d $F30KDIR ]; then
    echo "creating $F30KDIR"
    mkdir $F30KDIR
fi
if [ ! -d $F30KDIR/flickr30k_entities ]; then
    git clone https://github.com/BryanPlummer/flickr30k_entities.git $F30KDIR/flickr30k_entities
    unzip $F30KDIR/flickr30k_entities/annotations.zip -d $F30KDIR/flickr30k_entities
fi
if [ ! -d $F30KDIR/flicker/flickr30k-images ]; then
    tar -xvf $F30KDIR/flicker/flickr30k-images.tar.gz -C $F30KDIR/flicker
fi
if [ ! -f $F30KDIR/flicker.zip ]; then
    tar -xvf $F30KDIR/flicker/flickr30k-images.tar.gz -C $F30KDIR/flicker
fi


# refcoco/refcocog/refcocop dataset
# refer to https://github.com/jackroos/VL-BERT
REFCOCODIR=$DIR"/refcoco"
if [ ! -d $REFCOCODIR ]; then
    echo "creating $REFCOCODIR"
    mkdir $REFCOCODIR
    ln -s $PWD/$COCODIR/train2014 $REFCOCODIR/train2014
    ln -s $PWD/$COCODIR/val2014 $REFCOCODIR/val2014
    ln -s $PWD/$VQADIR/test2015 $REFCOCODIR/test2015
fi
if [ ! -f $REFCOCODIR"/refcoco.zip" ]; then
    wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip -P $REFCOCODIR
    unzip $REFCOCODIR/refcoco.zip -d $REFCOCODIR
fi
if [ ! -f $REFCOCODIR"/refcoco+.zip" ]; then
    wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip -P $REFCOCODIR
    unzip $REFCOCODIR/refcoco+.zip -d $REFCOCODIR
fi
if [ ! -f $REFCOCODIR"/refcocog.zip" ]; then
    wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip -P $REFCOCODIR
    unzip $REFCOCODIR/refcocog.zip -d $REFCOCODIR
fi
if [ ! -f $REFCOCODIR"/refclef.zip" ]; then
    wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip -P $REFCOCODIR
    unzip $REFCOCODIR/refclef.zip -d $REFCOCODIR
fi
if [ ! -f $REFCOCODIR"/annotations_trainval2014.zip" ]; then
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P $REFCOCODIR
    unzip $REFCOCODIR/annotations_trainval2014.zip -d $REFCOCODIR
fi
if [ ! -f $REFCOCODIR"/image_info_test2015.zip" ]; then
    wget http://images.cocodataset.org/annotations/image_info_test2015.zip -P $REFCOCODIR
    unzip $REFCOCODIR/image_info_test2015.zip -d $REFCOCODIR
fi


# GQA dataset
GQADIR=$DIR"/GQA"
if [ ! -d $GQADIR ]; then
    echo "creating $GQADIR"
    mkdir $GQADIR
fi
if [ ! -f $GQADIR"/images.zip" ]; then
    wget https://nlp.stanford.edu/data/gqa/images.zip -P $GQADIR
    unzip $GQADIR/images.zip -d $GQADIR
fi

# mdetr annotatios
MDETRDIR=$DIR"/MDETR"
if [ ! -d $MDETRDIR ]; then
    echo "creating $MDETRDIR"
    mkdir $MDETRDIR
fi
if [ ! -f $MDETRDIR"/mdetr_annotations.tar.gz?download=1" ]; then
    wget https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1 -P $MDETRDIR
    tar -xvf $MDETRDIR"/mdetr_annotations.tar.gz?download=1" -C $MDETRDIR
    mv $MDETRDIR/OpenSource $MDETRDIR/mdetr_annotations
fi


# COPSREF annotatios
COPSREF=$DIR"/COPSREF"
if [ ! -d $COPSREF ]; then
    echo "creating $COPSREF"
    mkdir $COPSREF
fi
# Download the copsref json file from https://github.com/zfchenUnique/Cops-Ref
# Place the json file under $COPSREF"/Cops-Ref.json"


# ImageClef annotatios for RefClef (ReferItGame)
IMAGECLEF=$DIR"/IMAGECLEF"
if [ ! -d $IMAGECLEF ]; then
    echo "creating $IMAGECLEF"
    mkdir $IMAGECLEF
fi
if [ ! -f $IMAGECLEF"/iaprtc12.tgz" ]; then
    wget http://www-i6.informatik.rwth-aachen.de/imageclef/resources/iaprtc12.tgz -P $IMAGECLEF
    tar -xvzf $IMAGECLEF"/iaprtc12.tgz" -C $IMAGECLEF
fi


