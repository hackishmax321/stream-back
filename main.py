from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pathlib import Path
import uvicorn
from PIL import Image
import numpy as np
import tensorflow as tf
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
from bson import ObjectId
import bcrypt
import logging
import os
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo import ReturnDocument


app = FastAPI()
origins = [
    "https://stream-ui-9g0h.onrender.com",
    "http://localhost:3000",
    "http://localhost:3001"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = "./uploads"
FACES_DIR = "./faces"

Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)


# Setting up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# DB Configurations -> sub-stream
MONGODB_CONNECTION_URL = "mongodb+srv://subuser:111222333@cluster0.jg2xlsc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGODB_CONNECTION_URL)
db = client["edu_db"]
user_collection = db["users"]
student_collection = db["students"]
fines_collection = db["fines"]
enrollments_collection = db["enrollments"]


class StudentModel(BaseModel):
    student_id: str  # Student ID (unique identifier)
    email: EmailStr
    password: str
    role: str = "student"  # Default role
    avatar: str = ""  # URL to profile picture
    first_name: str
    last_name: str
    contact_number: str
    address: str
    gender: str
    date_of_birth: str  # Added for student context
    department: str  # Added for student context
    enrollment_date: str  # Added for student context

class StudentResponseModel(BaseModel):
    id: str
    student_id: str
    email: EmailStr
    role: str
    avatar: str
    first_name: str
    last_name: str
    contact_number: str
    address: str
    gender: str
    date_of_birth: str
    department: str
    enrollment_date: str

# Database operations
async def get_student_by_email(email: str):
    return await student_collection.find_one({"email": email})

async def get_student_by_id(student_id: ObjectId):
    return await student_collection.find_one({"_id": student_id})

async def get_student_by_student_id(student_id: str):
    return await student_collection.find_one({"student_id": student_id})

# Student Endpoints
@app.post("/students", response_model=StudentResponseModel)
async def create_student(student: StudentModel):
    # Check if student already exists by student_id or email
    student_exists = await get_student_by_email(student.email)
    if student_exists:
        raise HTTPException(status_code=400, detail="Student with this ID or email already exists")
    
    # Hash the student's password
    hashed_password = bcrypt.hashpw(student.password.encode('utf-8'), bcrypt.gensalt())
    student.password = hashed_password.decode('utf-8')
    
    # Insert the student into the database
    student_dict = jsonable_encoder(student)
    result = await student_collection.insert_one(student_dict)
    
    # Retrieve the inserted student
    new_student = await get_student_by_id(result.inserted_id)
    
    # Prepare response
    return StudentResponseModel(
        id=str(new_student["_id"]),
        student_id=new_student["student_id"],
        email=new_student["email"],
        role=new_student["role"],
        avatar=new_student["avatar"],
        first_name=new_student["first_name"],
        last_name=new_student["last_name"],
        contact_number=new_student["contact_number"],
        address=new_student["address"],
        gender=new_student["gender"],
        date_of_birth=new_student["date_of_birth"],
        department=new_student["department"],
        enrollment_date=new_student["enrollment_date"]
    )

@app.get("/students/{student_id}", response_model=StudentResponseModel)
async def get_student(student_id: str):
    student = await get_student_by_student_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    return StudentResponseModel(
        id=str(student["_id"]),
        student_id=student["student_id"],
        email=student["email"],
        role=student["role"],
        avatar=student["avatar"],
        first_name=student["first_name"],
        last_name=student["last_name"],
        contact_number=student["contact_number"],
        address=student["address"],
        gender=student["gender"],
        date_of_birth=student["date_of_birth"],
        department=student["department"],
        enrollment_date=student["enrollment_date"]
    )

@app.get("/students/by-email/{email}", response_model=StudentResponseModel)
async def get_studentdata_by_email(email: str):
    student = await get_student_by_email(email)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    return StudentResponseModel(
        id=str(student["_id"]),
        student_id=student["student_id"],
        email=student["email"],
        role=student["role"],
        avatar=student["avatar"],
        first_name=student["first_name"],
        last_name=student["last_name"],
        contact_number=student["contact_number"],
        address=student["address"],
        gender=student["gender"],
        date_of_birth=student["date_of_birth"],
        department=student["department"],
        enrollment_date=student["enrollment_date"]
    )

@app.get("/students", response_model=list[StudentResponseModel])
async def get_all_students():
    students_cursor = student_collection.find({})
    students = await students_cursor.to_list(None)

    return [
        StudentResponseModel(
            id=str(student["_id"]),
            student_id=student["student_id"],
            email=student["email"],
            role=student["role"],
            avatar=student["avatar"],
            first_name=student["first_name"],
            last_name=student["last_name"],
            contact_number=student["contact_number"],
            address=student["address"],
            gender=student["gender"],
            date_of_birth=student["date_of_birth"],
            department=student["department"],
            enrollment_date=student["enrollment_date"]
        ) for student in students
    ]

# Authentication (Student-specific)
SECRET_KEY = "x-ky0sce"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="student-login")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    student_id: Optional[str] = None
    email: Optional[str] = None

def verify_password(plain_password: str, hashed_password: str):
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

async def authenticate_student(student_id_or_email: str, password: str):
    # Try to find student by student_id first
    student = await get_student_by_student_id(student_id_or_email)
    if not student:
        # If not found by student_id, try by email
        student = await get_student_by_email(student_id_or_email)
        if not student:
            return False
    
    if not verify_password(password, student["password"]):
        return False
    
    return student

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/student-login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    student = await authenticate_student(form_data.username, form_data.password)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect Student ID/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": student["student_id"], "email": student["email"]}, 
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/student-session", response_model=StudentResponseModel)
async def read_student_me(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        student_id: str = payload.get("sub")
        email: str = payload.get("email")
        if student_id is None or email is None:
            raise credentials_exception
        token_data = TokenData(student_id=student_id, email=email)
    except JWTError:
        raise credentials_exception
    
    student = await get_student_by_student_id(token_data.student_id)
    if student is None:
        raise credentials_exception
    
    return StudentResponseModel(
        id=str(student["_id"]),
        student_id=student["student_id"],
        email=student["email"],
        role=student["role"],
        avatar=student["avatar"],
        first_name=student["first_name"],
        last_name=student["last_name"],
        contact_number=student["contact_number"],
        address=student["address"],
        gender=student["gender"],
        date_of_birth=student["date_of_birth"],
        department=student["department"],
        enrollment_date=student["enrollment_date"]
    )

#  FOr FInes

class FineModel(BaseModel):
    nic: str  # National Identity Card of the offender
    amount: float  # Fine amount
    title: str  # Reason for the fine (e.g., "Speeding", "Illegal Parking")
    date: str  # Date of the fine in string format (YYYY-MM-DD)
    station: str  # Police station that issued the fine
    more_info: Optional[str] = None  # Additional information
    status: str = "unpaid"  # Payment status: unpaid/paid/contested
    
    class Config:
        json_encoders = {ObjectId: str}

class FineResponseModel(FineModel):
    id: str  # This will be the string representation of MongoDB's _id

class FineUpdateModel(BaseModel):
    amount: Optional[float] = None
    title: Optional[str] = None
    date: Optional[str] = None
    station: Optional[str] = None
    more_info: Optional[str] = None
    status: Optional[str] = None

    
@app.post("/fines", response_model=FineResponseModel)
async def create_fine(fine: FineModel):
    fine_dict = fine.dict()
    result = await fines_collection.insert_one(fine_dict)
    created_fine = await fines_collection.find_one({"_id": result.inserted_id})
    return {**created_fine, "id": str(created_fine["_id"])}

@app.get("/fines", response_model=list[FineResponseModel])
async def get_all_fines():
    fines = []
    async for fine in fines_collection.find():
        fine["id"] = str(fine["_id"])
        fines.append(fine)
    return fines

@app.get("/fines/{nic}", response_model=list[FineResponseModel])
async def get_fines_by_nic(nic: str):
    fines = []
    async for fine in fines_collection.find({"nic": nic}):
        fine["id"] = str(fine["_id"])
        fines.append(fine)
    if not fines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No fines found for NIC: {nic}"
        )
    return fines

@app.get("/fines/id/{fine_id}", response_model=FineResponseModel)
async def get_fine_by_id(fine_id: str):
    fine = await fines_collection.find_one({"_id": ObjectId(fine_id)})
    if fine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fine with id {fine_id} not found"
        )
    fine["id"] = str(fine["_id"])
    return fine

@app.put("/fines/{fine_id}", response_model=FineResponseModel)
async def update_fine(fine_id: str, fine_update: FineUpdateModel):
    update_data = {k: v for k, v in fine_update.dict().items() if v is not None}
    
    if len(update_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No update data provided"
        )
    
    result = await fines_collection.update_one(
        {"_id": ObjectId(fine_id)},
        {"$set": update_data}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fine with id {fine_id} not found"
        )
    
    updated_fine = await fines_collection.find_one({"_id": ObjectId(fine_id)})
    updated_fine["id"] = str(updated_fine["_id"])
    return updated_fine

@app.delete("/fines/{fine_id}")
async def delete_fine(fine_id: str):
    result = await fines_collection.delete_one({"_id": ObjectId(fine_id)})
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fine with id {fine_id} not found"
        )
    return {"message": f"Fine {fine_id} deleted successfully"}

# Entrollment 

class EnrollmentBase(BaseModel):
    student_id: str
    email: EmailStr
    program: str
    enrollment_date: datetime = datetime.now()
    status: str = "active"  # active, completed, withdrawn
    academic_year: str
    semester: str

class EnrollmentCreate(EnrollmentBase):
    pass

class EnrollmentResponse(EnrollmentBase):
    id: str

    class Config:
        json_encoders = {ObjectId: str}


# Endpoints
@app.post("/enrollments", response_model=EnrollmentResponse, status_code=status.HTTP_201_CREATED)
async def create_enrollment(enrollment: EnrollmentCreate, token: str = Depends(oauth2_scheme)):
    # Check if student exists
    print(enrollment)
    student = await get_student_by_email(enrollment.email)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Check if email matches student record
    if student["email"] != enrollment.email:
        raise HTTPException(status_code=400, detail="Email doesn't match student record")
    
    # Check if student is already enrolled in this program
    existing_enrollment = await enrollments_collection.find_one({
        "student_id": enrollment.student_id,
        "program": enrollment.program,
        "status": "active"
    })
    if existing_enrollment:
        raise HTTPException(status_code=400, detail="Student is already enrolled in this program")
    
    # Create enrollment
    enrollment_dict = enrollment.dict()
    result = await enrollments_collection.insert_one(enrollment_dict)
    
    # Return the created enrollment
    created_enrollment = await enrollments_collection.find_one({"_id": result.inserted_id})
    return EnrollmentResponse(**created_enrollment, id=str(result.inserted_id))

@app.get("/enrollments", response_model=List[EnrollmentResponse])
async def get_all_enrollments(token: str = Depends(oauth2_scheme)):
    enrollments = []
    async for enrollment in enrollments_collection.find():
        enrollments.append(EnrollmentResponse(**enrollment, id=str(enrollment["_id"])))
    return enrollments

@app.get("/enrollments/{enrollment_id}", response_model=EnrollmentResponse)
async def get_enrollment_by_id(enrollment_id: str, token: str = Depends(oauth2_scheme)):
    if not ObjectId.is_valid(enrollment_id):
        raise HTTPException(status_code=400, detail="Invalid enrollment ID")
    
    enrollment = await enrollments_collection.find_one({"_id": ObjectId(enrollment_id)})
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    
    return EnrollmentResponse(**enrollment, id=str(enrollment["_id"]))

@app.get("/enrollments/student/{student_id}", response_model=List[EnrollmentResponse])
async def get_enrollments_by_student_id(student_id: str, token: str = Depends(oauth2_scheme)):
    enrollments = []
    async for enrollment in enrollments_collection.find({"student_id": student_id}):
        enrollments.append(EnrollmentResponse(**enrollment, id=str(enrollment["_id"])))
    
    if not enrollments:
        raise HTTPException(status_code=404, detail="No enrollments found for this student")
    
    return enrollments

@app.get("/enrollments/email/{email}", response_model=List[EnrollmentResponse])
async def get_enrollments_by_email(email: str, token: str = Depends(oauth2_scheme)):
    enrollments = []
    async for enrollment in enrollments_collection.find({"email": email}):
        enrollments.append(EnrollmentResponse(**enrollment, id=str(enrollment["_id"])))
    
    if not enrollments:
        raise HTTPException(status_code=404, detail="No enrollments found for this email")
    
    return enrollments

@app.put("/enrollments/{enrollment_id}", response_model=EnrollmentResponse)
async def update_enrollment(
    enrollment_id: str, 
    update_data: dict, 
    token: str = Depends(oauth2_scheme)
):
    if not ObjectId.is_valid(enrollment_id):
        raise HTTPException(status_code=400, detail="Invalid enrollment ID")
    
    # Remove None values from update data
    update_data = {k: v for k, v in update_data.items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No data provided for update")
    
    updated_enrollment = await enrollments_collection.find_one_and_update(
        {"_id": ObjectId(enrollment_id)},
        {"$set": update_data},
        return_document=ReturnDocument.AFTER
    )
    
    if not updated_enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    
    return EnrollmentResponse(**updated_enrollment, id=str(updated_enrollment["_id"]))

@app.delete("/enrollments/{enrollment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_enrollment(enrollment_id: str, token: str = Depends(oauth2_scheme)):
    if not ObjectId.is_valid(enrollment_id):
        raise HTTPException(status_code=400, detail="Invalid enrollment ID")
    
    result = await enrollments_collection.delete_one({"_id": ObjectId(enrollment_id)})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
