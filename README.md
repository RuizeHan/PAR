# PAR
Panoramic Human Activity Recognition

We have released the dataset and code to the public.

Dataset Link: https://pan.baidu.com/s/1K8RDNteaphYJY8YEAg5fyA Password: PHAR

[2022.11] We update the code at PAR-main.zip


[2023.04] We have updated the source code. We have also provided the individual action, group activity, and global activity categories with the corresponding IDs.

[2023.04] About the explanation of the group activity labels: The vector length of the social group activity label is 32. This is because when the model is dealing with social activity, a group with one person (i.e. a person who does not belong to any social group) is also considered. In this case, the group activity is assigned with the individual action label (27 categories). 
In our later study, we may update this annotation.


[2023.09] We upload the evaluation code of the group detection. 
We also update the code.

[2023.09] We upload the evaluation code of the group detection.


[2023.10] We uploaded the base model of stage I to the cloud storage. Put this file into the path ./data.

https://pan.baidu.com/s/1eW9uj7wO8vaFgWSoRD-UeA @ PHAR.

```
@inproceedings{han2022panoramic,
  title={Panoramic Human Activity Recognition},
  author={Han, Ruize and Yan, Haomin and Li, Jiacheng and Wang, Songmiao and Feng, Wei and Wang, Song},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part IV},
  pages={244--261},
  year={2022},
  organization={Springer}
}
```
