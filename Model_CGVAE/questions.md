460: why always tf.constant(0), nothing update at each iteration?  
        idx_final, cross_entropy_losses_final, edge_predictions_final,edge_type_predictions_final=\
            tf.while_loop(lambda idx, cross_entropy_losses,edge_predictions,edge_type_predictions: idx < self.placeholders['max_iteration_num'],
            self.generate_cross_entropy,
            (tf.constant(0), cross_entropy_losses,edge_predictions,edge_type_predictions,))


