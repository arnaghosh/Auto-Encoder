require 'image';

folder = "/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/FGVC/fgvc-aircraft-2013b/data/"

families = {};
family_file = folder.."families.txt";
family_f = io.open(family_file);
l=0;
for _ in io.lines(family_file) do
	families[_] = l;
	l=l+1;
end

variants = {};
variant_file = folder.."variants.txt";
variant_f = io.open(variant_file);
l=0;
for _ in io.lines(variant_file) do
	variants[_] = l;
	l=l+1;
end

manufacturers = {};
man_file = folder.."manufacturers.txt";
man_f = io.open(man_file);
l=0;
for _ in io.lines(man_file) do
	manufacturers[_] = l;
	l=l+1;
end


imNames_file = folder.."images_train.txt";
imNames_f = io.open(imNames_file);
l=0;
imNames = {};
for _ in io.lines(imNames_file) do
	l=l+1;
	--img = image.load(folder..'images/'.._..'.jpg');
	imNames[l] = _;
end

print(#imNames)

im_fam_file = folder.."images_family_train.txt";
im_fam_f = io.open(im_fam_file);
l=0;
im_fam = {};
for _ in io.lines(im_fam_file) do
	l=l+1;
	--img = image.load(folder..'images/'.._..'.jpg');
	x = _:split(" ");
	im_fam[l]=x[2];
	for i=3,#x do
		im_fam[l] = im_fam[l]..' '..x[i]; 
	end
	im_fam[l] = families[im_fam[l]]
	if im_fam[l]==nil then print("poota1") end
end

im_var_file = folder.."images_variant_train.txt";
im_var_f = io.open(im_var_file);
l=0;
im_var = {};
for _ in io.lines(im_var_file) do
	l=l+1;
	--img = image.load(folder..'images/'.._..'.jpg');
	x = _:split(" ");
	im_var[l]=x[2];
	for i=3,#x do
		im_var[l] = im_var[l]..' '..x[i]; 
	end
	im_var[l] = variants[im_var[l]]
	if im_var[l]==nil then print("poota2") end
end

im_man_file = folder.."images_manufacturer_train.txt";
im_man_f = io.open(im_man_file);
l=0;
im_man = {};
for _ in io.lines(im_man_file) do
	l=l+1;
	--img = image.load(folder..'images/'.._..'.jpg');
	x = _:split(" ");
	im_man[l]=x[2];
	for i=3,#x do
		im_man[l] = im_man[l]..' '..x[i]; 
	end
	im_man[l] = manufacturers[im_man[l]]
	if im_man[l]==nil then print("poota3") end
end

torch.save('train_im_labels.t7',{imNames,im_fam,im_var,im_man})
