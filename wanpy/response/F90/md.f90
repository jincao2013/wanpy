! *************************************
!  Program for Simple MD
!  
!  Coded by JinCao Apr.10, 2017
!  
! *************************************  

module md
    ! *************************************
    !  module : md
    !  
    !  Attention:
    !  Subroutine of level 2 and higher level is hidden for external program !
    !  
    !  ** Level 1 :
    !  initialize(L,N,N_per_dim,R,r_matrix,v_matrix,f_matrix)
    !  thermostat(L,N,Boltzmann,R,Tempture_Constant,v_matrix,t)
    !  update_position_velocity(L,N,R,Delta_t,r_matrix,v_matrix,f_matrix,f_matrix_new)
    !  io(N,md_step,r_matrix,v_matrix,f_matrix,t)
    !  
    !  ** Level 2 :
    !  cal_force(L,N,R,r_matrix,f_matrix)
    !  apply_boundary(L,N,r_matrix)
    !  
    !  ** Level 3 :
    !  gen_r_matrix_center(L,N,o,r_matrix,r_matrix_center)
    !  
    !  Debug Tools:
    !  print *,"Hello, i am here ..."
    !  
    ! *************************************    
    contains

    !  ** Level 1 :

    subroutine initialize(L,N,N_per_dim,R,r_matrix,v_matrix,f_matrix)
        implicit none

        ! Define Parameter
        ! integer :: L,N,N_per_dim
        ! parameter (L=10,N=64,N_per_dim = 4)
        integer, intent(in) :: N,N_per_dim
        real, intent(in) :: L,R

        ! Define in(out)put Variable
        real, intent(out), dimension(3,N) :: r_matrix,v_matrix,f_matrix

        ! Define Inner Variable
        integer :: i,j,k,count
        real :: random

        ! Subroutine
        count = 1
        do i=1,N_per_dim
            do j=1,N_per_dim
                do k=1,N_per_dim
                	! position
                    r_matrix(1,count) = (i-0.5)*L/N_per_dim
                    r_matrix(2,count) = (j-0.5)*L/N_per_dim
                    r_matrix(3,count) = (k-0.5)*L/N_per_dim
                    ! call random_number(random)
                    ! r_matrix(1,count) = random*L
                    ! call random_number(random)
                    ! r_matrix(2,count) = random*L
                    ! call random_number(random)
                    ! r_matrix(3,count) = random*L
                    ! velocity
                    call random_number(random)
                    v_matrix(1,count) = 2*random-1
                    call random_number(random)
                    v_matrix(2,count) = 2*random-1
                    call random_number(random)
                    v_matrix(3,count) = 2*random-1
                    count = count+1
                end do
            end do
        end do

        call cal_force(L,N,R,r_matrix,f_matrix)

        ! del '!' to test this subroutine
        !do i=1,N
        !    print *,i
        !    print *,"R",r_matrix(1,i),r_matrix(2,i),r_matrix(3,i)
        !    print *,"V",v_matrix(1,i),v_matrix(2,i),v_matrix(3,i)
        !    print *,"F",f_matrix(1,i),f_matrix(2,i),f_matrix(3,i)
        !end do
    end subroutine initialize

    subroutine thermostat(L,N,Boltzmann,R,Tempture_Constant,v_matrix,t)
        implicit none

        ! Define Parameter
        ! integer :: L,N,Boltzmann
        ! real :: R
        ! parameter (L=10,N=64,R=4.0,Boltzmann=1)
        integer, intent(in) :: N
        real, intent(in) :: Boltzmann
        real, intent(in) :: L,R,Tempture_Constant

        ! Define in(out)put Variable
        real, intent(inout), dimension(3,N) :: v_matrix
        real, intent(out) :: t

        ! Define Inner Variable
        integer i
        real beta

        ! Subroutine
        t=0
        do i=1,N
            t=t+v_matrix(1,i)**2 + &
                v_matrix(2,i)**2 + &
                v_matrix(3,i)**2
        end do
        t=t/(Boltzmann*(3*N-3))

        beta=(Tempture_Constant/t)**0.5

        do i=1,N
            v_matrix(1,i)=v_matrix(1,i)*beta
            v_matrix(2,i)=v_matrix(2,i)*beta
            v_matrix(3,i)=v_matrix(3,i)*beta
        end do

    end subroutine thermostat

    subroutine update_position_velocity(L,N,R,Delta_t,r_matrix,v_matrix,f_matrix,f_matrix_new)
        ! *************************************
        ! input : r_matrix,v_matrix,f_matrix
        ! output : r_matrix,v_matrix,f_matrix_new
        ! *************************************
        implicit none

        ! Define Parameter
        ! integer :: L,N
        ! parameter (L=10,N=64)
        integer, intent(in) :: N
        real, intent(in) :: L,R,Delta_t

        ! Define in(out)put Variable
        real, intent(inout), dimension(3,N) :: r_matrix,v_matrix
        real, intent(in), dimension(3,N) :: f_matrix
        real, intent(out), dimension(3,N) :: f_matrix_new
        
        ! Define Inner Variable
        real, dimension(3,N) :: r_matrix_new,v_matrix_new
        
        integer :: i,j

        ! Subroutine

        ! *************************************
        ! update position
        ! *************************************
        do i=1,N
            r_matrix_new(1,i)=r_matrix(1,i) + v_matrix(1,i)*Delta_t + &
                              0.5*f_matrix(1,i)*Delta_t**2
            r_matrix_new(2,i)=r_matrix(2,i) + v_matrix(2,i)*Delta_t + &
                              0.5*f_matrix(2,i)*Delta_t**2
            r_matrix_new(3,i)=r_matrix(3,i) + v_matrix(3,i)*Delta_t + &
                              0.5*f_matrix(3,i)*Delta_t**2
        end do
        call apply_boundary(L,N,r_matrix_new)

        ! *************************************
        ! update velocity
        ! *************************************
        call cal_force(L,N,R,r_matrix_new,f_matrix_new)
        do i=1,N
            v_matrix_new(1,i)=v_matrix(1,i) + &
                               0.5*Delta_t*(f_matrix(1,i)+f_matrix_new(1,i))
            v_matrix_new(2,i)=v_matrix(2,i) + &
                               0.5*Delta_t*(f_matrix(2,i)+f_matrix_new(2,i))
            v_matrix_new(3,i)=v_matrix(3,i) + &
                               0.5*Delta_t*(f_matrix(3,i)+f_matrix_new(3,i))
        end do

        ! *************************************
        ! return r_matrix and v_matrix
        ! *************************************
        do i=1,3
            do j=1,N
                r_matrix(i,j)=r_matrix_new(i,j)
                v_matrix(i,j)=v_matrix_new(i,j)
            end do
        end do

    end subroutine update_position_velocity

    !subroutine cal_distance(r_matrix,distance_matrix)
    !    implicit none
    !
    !    integer :: L,N
    !    parameter (L=10,N=64)
    !    integer :: i,j,k
    !
    !    real, intent(in), dimension(3,N) :: r_matrix
    !    real, intent(out), dimension(N,N) :: distance_matrix
    !    real, dimension(3,N) :: r_matrix_center
    !
    !    do i=1,N
    !        do j=i,N
    !            if i==j then
    !                distance_matrix(i,j)=0
    !            else
    !                call gen_r_matrix_center(i,r_matrix,r_matrix_center)
    !                distance_matrix(i,j)=(r_matrix_center(1,j)**2 + &
    !                                      r_matrix_center(2,j)**2 + &
    !                                      r_matrix_center(3,j)**2)**0.5
    !                distance_matrix(j,i)=distance_matrix(i,j)
    !            end if
    !        end do
    !    end do
    !end subroutine cal_distance

    subroutine io(N,md_step,r_matrix,v_matrix,f_matrix,t)
        implicit none

        ! Define Parameter
        integer, intent(in) :: N

        ! Define in(out)put Variable
        integer, intent(in) :: md_step
        real, intent(in), dimension(3,N) :: r_matrix,v_matrix,f_matrix
        real, intent(in) :: t

        ! Define Inner Variable
        integer :: i
        integer :: status             ! I/O status

        ! Subroutine
        open (UNIT=1, FILE='md.out', STATUS='REPLACE', ACTION='WRITE', &
                IOSTAT=status )

        write (1,100) md_step
        100 format ("** MD step ",I10)

        write (1,*) "Tempture ",t

        write (1,*) "Position Matrix"
        do i=1,N
            write (1,200) i,r_matrix(1,i),r_matrix(2,i),r_matrix(3,i)
            200 format (I3,T7,F10.6,3X,F10.6,3X,F10.6,3X)
        end do

        write (1,*) "Velocity Matrix"
        do i=1,N
            write (1,300) i,v_matrix(1,i),v_matrix(2,i),v_matrix(3,i)
            300 format (I3,T7,F10.6,3X,F10.6,3X,F10.6,3X)
        end do

        write (1,*) "Force Matrix"
        do i=1,N
            write (1,400) i,f_matrix(1,i),f_matrix(2,i),f_matrix(3,i)
            400 format (I3,T7,F16.6,3X,F16.6,3X,F16.6,3X)
        end do

        write (1,*) ""
        write (1,*) ""


    end subroutine io

    !  ** Level 2 :

    subroutine cal_force(L,N,R,r_matrix,f_matrix)
        implicit none

        ! Define Parameter
        ! integer :: L,N
        ! real :: R
        ! parameter (L=10,N=64,R=4.0)
        integer, intent(in) :: N
        real, intent(in) :: L,R
        
        ! Define in(out)put Variable
        real, intent(in), dimension(3,N) :: r_matrix
        real, intent(out), dimension(3,N) :: f_matrix

        ! Define Inner Variable
        integer :: i,j
        real, dimension(3,N) :: r_matrix_center
        real :: distance,temp

        ! Subroutine
        do i=1,3
            do j=1,N
                f_matrix(i,j)=0
            end do
        end do
        
        do i=1,N
            call gen_r_matrix_center(L,N,i,r_matrix,r_matrix_center)
            
            do j=1,N
                if ( i==j ) then 
                    cycle
                end if
                distance=(r_matrix_center(1,j)**2 + &
                          r_matrix_center(2,j)**2 + &
                          r_matrix_center(3,j)**2)**0.5
                temp=48*(1/distance**14-0.5/distance**8)
                if ( distance < R ) then
                    f_matrix(1,i)=f_matrix(1,i)+r_matrix_center(1,j)*temp
                    f_matrix(2,i)=f_matrix(2,i)+r_matrix_center(2,j)*temp
                    f_matrix(3,i)=f_matrix(3,i)+r_matrix_center(3,j)*temp
                    ! print *,distance,temp,r_matrix_center(1,j),r_matrix_center(2,j),r_matrix_center(3,j)
                end if
            end do

            ! if (f_matrix(1,i)>100) then 
            !     print *,"f=",f_matrix(1,i),f_matrix(2,i),f_matrix(3,i)
            !     print *,""
            !     print *,""
            ! end if 

        end do

    end subroutine cal_force

    subroutine apply_boundary(L,N,r_matrix)
        implicit none

        ! Define Parameter
        ! integer :: L,N
        ! parameter (L=10,N=64)
        integer, intent(in) :: N
        real, intent(in) :: L

        ! Define in(out)put Variable
        real, intent(inout), dimension(3,N) :: r_matrix

        ! Define Inner Variable
        integer :: i,j

        ! Subroutine
        do i=1,3
            do j=1,N
                if ( r_matrix(i,j)>L ) then
                    r_matrix(i,j)=r_matrix(i,j)-L
                    if ( r_matrix(i,j)>L ) then
                        print *,"Warning: Too Large Force ! "
                        print *,"MD program stop !"
                        stop
                    end if
                else if ( r_matrix(i,j)<0 ) then
                    r_matrix(i,j)=r_matrix(i,j)+L
                    if ( r_matrix(i,j)<0 ) then
                        print *,"Warning: Too Large Force ! "
                        print *,"MD program stop !"
                        stop
                    end if
                end if
            end do 
        end do
    end subroutine apply_boundary

    !  ** Level 3 :

    subroutine gen_r_matrix_center(L,N,o,r_matrix,r_matrix_center)
        implicit none

        ! Define Parameter
        ! integer :: L,N
        ! parameter (L=10,N=64)
        integer, intent(in) :: N
        real, intent(in) :: L

        ! Define in(out)put Variable
        real, intent(in), dimension(3,N) :: r_matrix
        real, intent(out), dimension(3,N) :: r_matrix_center

        ! Define Inner Variable
        integer :: i,j
        integer :: o

        ! Subroutine
        do j=1,N
            ! r_matrix_center(1,j)=r_matrix(1,j)-r_matrix(1,o)
            ! r_matrix_center(2,j)=r_matrix(2,j)-r_matrix(2,o)
            ! r_matrix_center(3,j)=r_matrix(3,j)-r_matrix(3,o)
            do i=1,3
                r_matrix_center(i,j)=r_matrix(i,j)-r_matrix(i,o)
                if ( r_matrix_center(i,j) > L/2 ) then
                    r_matrix_center(i,j)=r_matrix_center(i,j)-L
                else if ( r_matrix_center(i,j) < -L/2 ) then
                    r_matrix_center(i,j)=r_matrix_center(i,j)+L
                end if
            end do
        end do

    end subroutine gen_r_matrix_center

end module md


program solve_md
    use md
    implicit none
    ! Define Parameter
    integer :: N,N_per_dim
    integer :: Steps
    real :: L,Delta_t,Tempture_Constant,R,Boltzmann
    parameter (N=64,N_per_dim=4, &
               Steps=10000, &
               L=10.0,Delta_t=0.1,Tempture_Constant=0.85,R=4.0,Boltzmann=0.1)

    ! Define Variable
    real, dimension(3,N) :: r_matrix,v_matrix,f_matrix,f_matrix_new
    integer :: i,j,md_step
    real :: t

    ! main

    !call initialize(L,N,N_per_dim,R,r_matrix,v_matrix,f_matrix)
    !call thermostat(L,N,Boltzmann,R,Tempture_Constant,v_matrix)
    !call update_position_velocity(L,N,R,r_matrix,v_matrix,f_matrix,f_matrix_new)
    !
    !do i=1,N
    !    print *,i
    !    print *,"R",r_matrix(1,i),r_matrix(2,i),r_matrix(3,i)
    !    print *,"V",v_matrix(1,i),v_matrix(2,i),v_matrix(3,i)
    !    print *,"F",f_matrix(1,i),f_matrix(2,i),f_matrix(3,i)
    !    print *,"F`",f_matrix_new(1,i),f_matrix_new(2,i),f_matrix_new(3,i)
    !end do


    call initialize(L,N,N_per_dim,R,r_matrix,v_matrix,f_matrix)
    call thermostat(L,N,Boltzmann,R,Tempture_Constant,v_matrix,t)
    call io(N,0,r_matrix,v_matrix,f_matrix,t)

    do md_step=1,Steps
    	call thermostat(L,N,Boltzmann,R,Tempture_Constant,v_matrix,t)
        call update_position_velocity(L,N,R,Delta_t,r_matrix,v_matrix,f_matrix,f_matrix_new)

        do i=1,3
            do j=1,N
                f_matrix(i,j)=f_matrix_new(i,j)
            end do 
        end do

        call io(N,md_step,r_matrix,v_matrix,f_matrix,t)

    end do

end program solve_md



