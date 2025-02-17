import numpy as np


class StopCriterion:
    def __init__(self, criteria, patience=5, loss_window=5):
        """
        Initialize the stopping criterion.

        Args:
            criteria (list): List of stopping criteria to use. Options: ['max_epochs', 'early_stopping', 'loss_plateau']
            patience (int): Number of epochs to wait before stopping if no improvement.
            loss_window (int): Number of epochs to track for loss plateau detection.
        """
        self.criteria = criteria
        self.max_epochs = None
        self.patience = patience
        self.loss_window = loss_window

        # Early Stopping
        self.best_loss = float('inf')
        self.waited_epochs = 0
        self.best_weights = None

        # Loss Tracking
        self.train_losses = []
        self.val_losses = []

    def set_max_epochs(self, max_epochs):
        """
        Set the maximum number of epochs.
        """
        self.max_epochs = max_epochs

    def check_early_stopping(self, val_loss, weights):
        """
        Check if early stopping criterion is met (no improvement for 'patience' epochs).
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.waited_epochs = 0
            self.best_weights = weights
        else:
            self.waited_epochs += 1
            if self.waited_epochs >= self.patience:
                return True, 'Early stopping', self.best_weights
        return False, None, weights

    def check_loss_plateau(self):
        """
        Check if the loss plateau criterion is met.
        """
        if len(self.train_losses) < self.loss_window:
            return False, None, None  # Not enough epochs to check

        recent_losses = self.train_losses[-self.loss_window:]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)

        if std_loss < (mean_loss * 0.001):  # Threshold for plateau
            return True, 'Loss plateau detected', None
        return False, None, None

    def check_max_epochs(self, current_epoch):
        """
        Check if the maximum number of epochs is reached.
        """
        if self.max_epochs and current_epoch >= self.max_epochs:
            return True, 'Maximum epochs reached', None
        return False, None, None

    def __call__(self, current_epoch, train_loss, val_loss, weights):
        """
        Check all stopping criteria and return whether to stop training.

        Args:
            current_epoch (int): The current epoch number.
            train_loss (float): The current training loss.
            val_loss (float): The current validation loss.
            weights (numpy array): The current model weights.

        Returns:
            (bool, str, numpy array): (stop_flag, stopping_reason, best_weights)
        """
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
