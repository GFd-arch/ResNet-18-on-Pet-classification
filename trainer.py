# initialize params and train / evaluate
import torch
import swanlab


class Trainer:
    def __init__(self, model, train_data, val_data, criterion, optimizer, scheduler, device=torch.device('cpu')):

        self.model = model.to(device)

        self.train_data = train_data
        self.val_data = val_data

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device

    def validate(self):

        self.model.eval()
        total_loss = 0.0

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_data:

                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_data)
        acc = correct / total
        self.model.train()

        return avg_loss, acc


    def train(self, epochs, val_interval=50):

        self.model.train()
        print("Start training...")
        global_step = 0
        for e in range(epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(self.train_data):

                global_step += 1
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # train log
                swanlab.log({
                    "train/loss": loss.item()
                })

                # print
                if (i + 1) % 10 == 0:
                    print(
                        f'Epoch [{e+1}/{epochs}] '
                        f'Step [{i+1}/{len(self.train_data)}] '
                        f'Loss: {running_loss / 10:.4f}'
                    )

                    running_loss = 0.0

                # validation
                if global_step % val_interval == 0:

                    val_loss, val_acc = self.validate()

                    print(
                        f'[Validation] '
                        f'Step {global_step} '
                        f'Val Loss: {val_loss:.4f} '
                        f'Val Acc: {val_acc:.4f}'
                    )

                    swanlab.log({
                        "val/loss": val_loss,
                        "val/acc": val_acc
                    })

            self.scheduler.step()


class Evaluator:
    def __init__(self, model, test_data, device=torch.device('cpu')):
        self.model = model.to(device)
        self.test_data = test_data
        self.device = device

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        print("Start evaluation...")
        with torch.no_grad():
            for images, labels in self.test_data:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')
        return accuracy
    
