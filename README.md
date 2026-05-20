# Random Forest Gump
Facial recognition project for an Introduction to Machine Learning course. Given a query face image, the system retrieves the most similar image of the same person from a gallery using face embeddings and similarity search.




comandi da scrivere su terminale macchina virtuale

crop_faces.py:
python crop_faces.py --input /percorso_cartella --output /percorso_cartella_destinazione(nuova, da creare)


per trainare (per mac):
cd ~/ML_project

python -u simclr_train.py \
  --data-folder ~/percorso_cartella_foto(deve essere cartella piatta) \
  --resume primo checkpoint per sequenziale (opzionale, se non c'è rimuovere).pt \
  --output nome_checkpoint_output.pt \
  --epochs 10 \
  --batch-size 256 (dipende dal numero di foto, aggiustare) \
  --margin 0.3 (si può cambiare, ma tendenzialmente va bene) \
  --freeze-stage-epochs (scrivere 0 se sequenziale, 3 se è il primo) \
  --workers 8 \
  --log-every 1
  
tail -f training_triplet.log (per vedere a che punto si è)



per la competition (per mac):
Bisogna cambiare il checkpoint con il nome del checkpoint che usiamo

python run_competition.py \
    --data-folder /percorso \
    --group-name "random_forest_group" \
    --checkpoint simclr_checkpoint.pt \ 
    --dry-run


