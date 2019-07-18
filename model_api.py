from peewee import *
from datetime import datetime, timedelta

db = SqliteDatabase('inference.db')

class BaseModel(Model):
    updatedAt = DateTimeField(default=datetime.now)
    createdAt = DateTimeField(default=datetime.now)

    def save(self, *args, **kwargs):
        self.updatedAt = datetime.now()
        return super(BaseModel, self).save(*args, **kwargs)

    class Meta:
        database = db

class Inference(BaseModel):
    video_url = CharField()
    input_file = CharField()
    output_file = CharField()
    status = IntegerField()
    result = CharField()

    class Meta:
        db_table = 'inference'