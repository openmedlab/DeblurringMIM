import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def save_training_plots(history, save_file_name):
    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(save_file_name + '_model_accuracy.jpg')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(save_file_name + '_model_loss.jpg')
    plt.close()



def save_training_plots_by_key(history, save_file_name, train_key, val_key):
    print(train_key, val_key)

    plt.plot(history.history[train_key])
    plt.plot(history.history[val_key])
    plt.title('model ' + train_key)
    plt.ylabel(train_key)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(save_file_name + '_model_' + train_key + '.jpg')
    plt.close()


def save_history_loss(train_hist, val_hist, save_file_name):
#   plt.plot(train_hist['loss'], '--')
    plt.plot(train_hist['loss_seg'], '--')
    plt.plot(train_hist['loss_cls'], '--')
  #  plt.plot(val_hist['loss'])
    plt.plot(val_hist['loss_seg'])
    plt.plot(val_hist['loss_cls'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_seg', 'train_cls', 'val_seg', 'val_cls'], loc='upper right')
    plt.savefig(save_file_name + '_model_loss.jpg')
    plt.close()


def save_history_acc(train_hist, val_hist, save_file_name):
    plt.plot(train_hist['dice'], '--')
    plt.plot(train_hist['f1'], '--')
    plt.plot(val_hist['dice'])
    plt.plot(val_hist['f1'])
    plt.title('model DICE & Accuracy & F1')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_dice', 'train_f1', 'val_dice', 'val_f1'], loc='upper right')
    plt.savefig(save_file_name + '_model_acc.jpg')
    plt.close()


def save_loss_f1(train_hist, val_hist, save_file_name):
    # summarize history for accuracy
    plt.plot(train_hist['loss'], '--')
    plt.plot(val_hist['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train - loss', 'val - loss'], loc='lower right')
    plt.savefig(save_file_name + '_model_loss.jpg')
    plt.close()

    plt.plot(train_hist['f1'], '--')
    plt.plot(val_hist['f1'])
    plt.title('model F1')
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.legend(['train - f1', 'val - f1'], loc='lower right')
    plt.savefig(save_file_name + '_model_f1.jpg')
    plt.close()



def save_loss_dice(train_hist, val_hist, save_file_name):
    # summarize history for accuracy
    plt.plot(train_hist['loss'], '--')
    plt.plot(val_hist['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train - loss', 'val - loss'], loc='lower right')
    plt.savefig(save_file_name + '_model_loss.jpg')
    plt.close()

    plt.plot(train_hist['dice'], '--')
    plt.plot(val_hist['dice'])
    plt.title('model DICE')
    plt.ylabel('DICE')
    plt.xlabel('epoch')
    plt.legend(['train - DICE', 'val - DICE'], loc='lower right')
    plt.savefig(save_file_name + '_model_dice.jpg')
    plt.close()

