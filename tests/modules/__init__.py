import os 
import torch


def test_model_rotations(model: torch.nn.Module, N: int = 24, M: int = 2000, checkpoint_path: str = None):
    # evaluate the `model` on N rotated versions of the first M images in the test set

    if checkpoint_path is not None:
        checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_path)

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        accuracies = np.load(checkpoint_path)
        return accuracies.tolist()

    model.eval()

    # to reduce interpolation artifacts (e.g. when testing the model on rotated images),
    # we upsample an image by a factor of 3, rotate it and finally downsample it again
    resize1 = Resize(87) # to upsample
    resize2 = Resize(29) # to downsample

    totensor = ToTensor()

    accuracies = []
    with torch.no_grad():
        model.eval()

        for r in tqdm(range(N)):
            total = 0
            correct = 0

            for i in range(M):
                x, t = raw_mnist_test[i]

                x = Image.fromarray(x.numpy()[0], mode='F')

                x = totensor(resize2(resize1(x).rotate(r*360./N, Image.BILINEAR))).reshape(1, 1, 29, 29).to(device)

                x = x.to(device)

                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += 1
                correct += (prediction == t).sum().item()

            accuracies.append(correct/total*100.)

    if checkpoint_path is not None:
        np.save(checkpoint_path, np.array(accuracies))

    return accuracies