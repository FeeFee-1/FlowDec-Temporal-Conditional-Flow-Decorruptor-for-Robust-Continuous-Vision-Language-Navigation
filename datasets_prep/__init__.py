import torchvision.transforms as transforms
from .sequence_dataset import SequenceDatasetWithFirstFrame

def get_corruption_dataset(args, augment=False, mixing_images1=None, mixing_images2=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = SequenceDatasetWithFirstFrame(
        image_path=args.image_path,
        action_path=args.action_path,
        transform=transform,
        keep_aspect_ratio=args.keep_aspect_ratio,
        augment=augment,
        mixing_images1=mixing_images1,
        mixing_images2=mixing_images2
    )

    return dataset




