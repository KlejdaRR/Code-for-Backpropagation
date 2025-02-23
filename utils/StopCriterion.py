import numpy as np

class StopCriterion:
    def __init__(self, criteria, patience=30, loss_window=30):
        """Initializing the stopping criterion."""
        self.criteria = criteria
        self.max_epochs = None
        self.patience = patience #  The number of epochs to wait for improvement before stopping
        self.loss_window = loss_window # The number of recent epochs to consider for loss plateau detection
        self.best_loss = float('inf') # Tracks the best validation loss observed during training
        self.waited_epochs = 0 # Counts the number of epochs since the last improvement in validation loss
        self.best_weights = None # Stores the model weights corresponding to the best validation loss

        # Lists to store training and validation losses over epochs
        self.train_losses = []
        self.val_losses = []

    def set_max_epochs(self, max_epochs):
        """Setting the maximum number of epochs."""
        self.max_epochs = max_epochs

    # Early stopping stops training if the validation loss does not improve for a specified number of epochs (patience)
    def check_early_stopping(self, val_loss, weights):
        """Checking if early stopping criterion is met."""

        # If the current validation loss (val_loss) is better than the best observed loss (self.best_loss),
        # we update self.best_loss and reset self.waited_epochs
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.waited_epochs = 0
            self.best_weights = weights
        else:
            # If the validation loss does not improve, we increment self.waited_epochs
            self.waited_epochs += 1

            # If self.waited_epochs exceeds self.patience, we stop training and return the best weights
            if self.waited_epochs >= self.patience:
                return True, 'Early stopping', self.best_weights
        return False, None, weights

    # checks if the validation loss has plateaued (stopped improving significantly) over a specified window of epochs
    def check_loss_plateau(self):
        """Checking if the loss plateau criterion is met."""

        # Ensuring there are enough epochs (self.loss_window) to evaluate the plateau
        if len(self.val_losses) < self.loss_window:
            return False, None, None

        # Computing the mean and standard deviation of the validation losses over the recent window
        recent_losses = self.val_losses[-self.loss_window:]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)

        # If the standard deviation is very small compared to the mean (less than 1% of the mean), the loss has plateaued
        if std_loss < (mean_loss * 0.01):
            return True, 'Loss plateau detected', None
        return False, None, None

    # checks if the maximum number of epochs (self.max_epochs) has been reached
    def check_max_epochs(self, current_epoch):
        """Checking if the maximum number of epochs is reached."""
        if self.max_epochs and current_epoch >= self.max_epochs:
            return True, 'Maximum epochs reached', None
        return False, None, None

    # checks all stopping criteria and determines whether to stop training
    def __call__(self, current_epoch, train_loss, val_loss, weights):
        """Checking all stopping criteria and return whether to stop training."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        for criterion in self.criteria:
            if criterion == 'max_epochs':
                stop, reason, best_weights = self.check_max_epochs(current_epoch)
                if stop:
                    return True, reason, best_weights

            elif criterion == 'early_stopping':
                stop, reason, best_weights = self.check_early_stopping(val_loss, weights)
                if stop:
                    return True, reason, best_weights

            elif criterion == 'loss_plateau':
                stop, reason, best_weights = self.check_loss_plateau()
                if stop:
                    return True, reason, best_weights

        return False, None, weights
