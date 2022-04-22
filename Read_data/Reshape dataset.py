import os

root_dir = 'hymenoptera_data\\val'
target_dir = 'bees_image'
img_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split('_')[0]
out_dir = "bees_label"
if not os.path.exists(os.path.join(root_dir, out_dir)):
    os.makedirs(os.path.join(root_dir, out_dir))
for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label)

print("success!")